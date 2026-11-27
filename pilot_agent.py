#!/usr/bin/env python3
"""
pilot_agent.py  -  pilot / main run for paper 2:
"Does an LLM agent add decision value over a calibrated cost rule?"

What it does
  1. loads the saved CalibRAG outputs (seed*_preds.npz) of ONE dataset and ONE seed,
  2. builds one prompt per test window (Agent-F arm, or Agent-F+R arm with the kNN evidence),
  3. queries an LLM (local OpenAI-compatible server, Anthropic API, or a mock for testing),
  4. parses  "DECISION: INTERVENE | WAIT"  (unparseable -> INTERVENE, and counted),
  5. writes every raw answer to a JSONL file (resumable) and prints the summary:
       parse rate, agreement with the rule, mean cost of agent vs rule, paired bootstrap CI.

The test set of C-MAPSS has ONE window per test engine, so a bootstrap over windows is a
bootstrap over engines. N-CMAPSS has many windows per unit: the bootstrap resamples whole units
and warns when there are fewer than 10 of them.

Examples (PowerShell, one line each)
  # 1. look at the prompts only, no model call
  python pilot_agent.py --results C:\\calibrag_results\\run_20261107_1613 --dataset FD001 --n 5 --dry_run
  # 2. test the whole code path without any model
  python pilot_agent.py --results C:\\calibrag_results\\run_20261107_1613 --dataset FD001 --n 50 --backend mock
  # 3. pilot on a local model served by vLLM / Ollama / llama.cpp (OpenAI-compatible)
  python pilot_agent.py --results ... --dataset FD001 --n 50 --backend openai --base_url http://localhost:8000/v1 --model <name>
  # 4. pilot on the Anthropic API (key in the environment variable ANTHROPIC_API_KEY)
  python pilot_agent.py --results ... --dataset FD001 --n 50 --backend anthropic --model <exact model id>
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import norm

Z80 = float(norm.ppf(0.90))  # same constant as calibrag_pipeline.py

# ------------------------------------------------------------------
# FROZEN PROMPT.  Do not edit after the pilot starts; the SHA-256 is saved with the results.
# ------------------------------------------------------------------
PROMPT_BASE = (
    "You are a maintenance decision assistant. A machine may fail within the next {H} cycles.\n\n"
    "Costs: intervening now costs {Cm}. Waiting costs {Cf} if the machine fails within {H} cycles, "
    "and 0 otherwise.\n\n"
    "A calibrated model gives: probability of failure within {H} cycles = {F}. "
    "Predicted remaining life (cycles): 10% quantile {Q10}, median {Q50}, 90% quantile {Q90}.\n"
    "{EVIDENCE}\n"
    "Decide whether to intervene now or wait. Explain briefly, then end with exactly one line: "
    "DECISION: INTERVENE or DECISION: WAIT."
)
# EXPLORATORY variant "noF" (added AFTER the main results were seen; must be labelled exploratory):
# identical wording, but the calibrated failure probability F is NOT given; only the quantiles are.
PROMPT_NOF = (
    "You are a maintenance decision assistant. A machine may fail within the next {H} cycles.\n\n"
    "Costs: intervening now costs {Cm}. Waiting costs {Cf} if the machine fails within {H} cycles, "
    "and 0 otherwise.\n\n"
    "A calibrated model predicts the remaining life (cycles): "
    "10% quantile {Q10}, median {Q50}, 90% quantile {Q90}.\n"
    "{EVIDENCE}\n"
    "Decide whether to intervene now or wait. Explain briefly, then end with exactly one line: "
    "DECISION: INTERVENE or DECISION: WAIT."
)
EVIDENCE_FR = ("Additional evidence: among the {K} most similar past windows, {NK} failed within "
               "the horizon.\n")
RULE_STATED = ("Rule: intervene if and only if (cost of failure) x (probability of failure) "
               "exceeds (cost of intervening).\n")  # only with --variant rule_stated

DECISION_RE = re.compile(r"DECISION\s*:\s*\**\s*(INTERVENE|WAIT)", re.IGNORECASE)


def prompt_hash(variant: str = "main") -> str:
    if variant == "noF":
        return hashlib.sha256(PROMPT_NOF.encode()).hexdigest()[:16]
    return hashlib.sha256((PROMPT_BASE + EVIDENCE_FR + RULE_STATED).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ data
def failure_prob(q10, q50, q90, H):
    sigma = np.maximum((q90 - q10) / (2 * Z80), 1e-6)
    return norm.cdf((H - q50) / sigma)


def load_preds(results, dataset, seed):
    pat = os.path.join(results, "**", dataset, f"seed{seed}_preds.npz")
    files = sorted(glob.glob(pat, recursive=True))
    if not files:
        sys.exit(f"No file matches {pat}")
    if len(files) > 1:
        print(f"[warn] {len(files)} files match, using the first: {files[0]}")
    d = np.load(files[0], allow_pickle=True)
    need = ["y", "q10C", "q50", "q90C", "H"]
    miss = [k for k in need if k not in d.files]
    if miss:
        sys.exit(f"{files[0]} lacks keys {miss}; available: {d.files[:40]}")
    H = int(d["H"])
    q10, q50, q90, y = (np.asarray(d[k], float) for k in ("q10C", "q50", "q90C", "y"))
    F = failure_prob(q10, q50, q90, H)
    if "F_C" in d.files and not np.allclose(F, d["F_C"], atol=1e-6):
        print("[warn] recomputed F differs from saved F_C (max abs diff "
              f"{np.max(np.abs(F - d['F_C'])):.2e}); using the recomputed one")
    knn = np.asarray(d["f_knn"], float) if "f_knn" in d.files else None
    unit = np.asarray(d["unit"]) if "unit" in d.files else np.arange(len(y))
    return dict(file=files[0], H=H, q10=q10, q50=q50, q90=q90, y=y, F=F, knn=knn, unit=unit)


def build_prompt(D, i, Cm, Cf, arm, variant, knn_k):
    ev = ""
    if arm == "FR":
        if D["knn"] is None:
            sys.exit("Arm FR needs 'f_knn' in the npz (the kNN retriever); not found.")
        ev = EVIDENCE_FR.format(K=knn_k, NK=int(round(D["knn"][i] * knn_k)))
    if variant == "rule_stated":
        ev = ev + RULE_STATED
    if variant == "noF":
        return PROMPT_NOF.format(H=D["H"], Cm=f"{Cm:g}", Cf=f"{Cf:g}", Q10=f"{D['q10'][i]:.2f}",
                                 Q50=f"{D['q50'][i]:.2f}", Q90=f"{D['q90'][i]:.2f}", EVIDENCE=ev)
    return PROMPT_BASE.format(H=D["H"], Cm=f"{Cm:g}", Cf=f"{Cf:g}", F=f"{D['F'][i]:.4f}",
                              Q10=f"{D['q10'][i]:.2f}", Q50=f"{D['q50'][i]:.2f}",
                              Q90=f"{D['q90'][i]:.2f}", EVIDENCE=ev)


def parse_decision(text):
    m = DECISION_RE.findall(text or "")
    if not m:
        return None
    return m[-1].upper()  # last occurrence


# ------------------------------------------------------------------ backends
class Backend:
    def __init__(self, a):
        self.a = a
        self.temp_ok = True
        self._tlock = threading.Lock()
        self._last = 0.0
        if a.backend == "openai":
            from openai import OpenAI
            self.c = OpenAI(base_url=a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        elif a.backend == "anthropic":
            import anthropic
            self.c = anthropic.Anthropic()
        elif a.backend == "mock":
            self.c = None
        else:
            sys.exit("unknown backend")

    def ask(self, prompt, seed_hint):
        a = self.a
        if a.backend == "mock":  # a noisy rule, only to test the code path
            mF = re.search(r"probability of failure within \d+ cycles = ([0-9]+(?:\.[0-9]+)?)", prompt)
            if mF:
                F = float(mF.group(1))
            else:   # noF variant: the mock behaves like a median-only agent (test of the code path only)
                H0 = float(re.search(r"within the next (\d+) cycles", prompt).group(1))
                med = float(re.search(r"median (-?[0-9]+(?:\.[0-9]+)?)", prompt).group(1))
                F = 1.0 if med < H0 else 0.0
            Cf = float(re.search(r"Waiting costs ([0-9]+(?:\.[0-9]+)?)", prompt).group(1))
            Cm = float(re.search(r"intervening now costs ([0-9]+(?:\.[0-9]+)?)", prompt).group(1))
            rng = np.random.RandomState(seed_hint % (2 ** 31))
            d = F * Cf > Cm
            if rng.rand() < 0.10:
                d = not d
            return f"Mock reasoning.\nDECISION: {'INTERVENE' if d else 'WAIT'}", "end_turn"
        kw = {}
        if getattr(a, "min_interval", 0.0) > 0:
            with self._tlock:                      # one request every min_interval seconds, all workers together
                wait = self._last + a.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
        if a.temperature is not None and self.temp_ok:
            kw["temperature"] = a.temperature
        for attempt in range(6):
            try:
                if a.backend == "anthropic":
                    r = self.c.messages.create(model=a.model, max_tokens=a.max_tokens,
                                               messages=[{"role": "user", "content": prompt}], **kw)
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), r.stop_reason
                extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else None
                r = self.c.chat.completions.create(model=a.model, max_tokens=a.max_tokens,
                                                   messages=[{"role": "user", "content": prompt}],
                                                   extra_body=extra, **kw)
                return r.choices[0].message.content, r.choices[0].finish_reason
            except Exception as e:  # noqa
                msg = str(e)
                if "temperature" in msg.lower() and "temperature" in kw:
                    print("[warn] this model rejects the temperature setting; retrying without it. "
                          "Write 'default temperature' in the paper.")
                    self.temp_ok = False
                    kw.pop("temperature")
                    continue
                if any(k in msg.lower() for k in ("credit balance", "authentication", "invalid x-api-key", "permission",
                                                  "per day", "(tpd)", "(rpd)", "daily", "invalid api key", "error code: 401", "per hour", "requests per hour")):
                    print(f"[FATAL] {msg[:200]}")
                    os._exit(3)  # stop the whole run now; finished answers are already saved
                wait = 2 ** attempt
                print(f"[retry {attempt + 1}/6 in {wait}s] {msg[:160]}")
                time.sleep(wait)
        return None, "error"


# ------------------------------------------------------------------ analysis
def cost(interv, fail, Cm, Cf):
    return np.where(interv, Cm, np.where(fail, Cf, 0.0))


def summarize(D, recs, a):
    H, Cm, Cf = D["H"], a.Cm, a.Cf
    tau = Cm / Cf
    fail_all = D["y"] <= H
    idxs = sorted({r["idx"] for r in recs})
    by = {i: [r for r in recs if r["idx"] == i] for i in idxs}
    n = len(idxs)
    F = D["F"][idxs]
    fail = fail_all[idxs]
    rule_i = F > tau
    rule_c = cost(rule_i, fail, Cm, Cf)
    agent_c, agree, unstable, parsed_ok, tot = np.zeros(n), np.zeros(n), np.zeros(n, bool), 0, 0
    for k, i in enumerate(idxs):
        ds = []
        for r in by[i]:
            tot += 1
            d = r["decision"]
            if d is not None:
                parsed_ok += 1
            ds.append(d != "WAIT")  # unparseable -> intervene
        ds = np.array(ds)
        agent_c[k] = np.mean(cost(ds, fail[k], Cm, Cf))
        agree[k] = np.mean(ds == rule_i[k])
        unstable[k] = len(set(ds.tolist())) > 1
    delta = agent_c - rule_c
    rng = np.random.RandomState(12345)
    units_i = D["unit"][idxs]
    uniq = np.unique(units_i)
    n_clusters = len(uniq)
    groups = [delta[units_i == u] for u in uniq]
    boots = np.empty(10000)
    for b in range(10000):
        pick = rng.randint(0, n_clusters, n_clusters)   # resample whole units (engines)
        boots[b] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if n_clusters < 10:
        print(f"[WARNING] only {n_clusters} independent units: the bootstrap interval below is NOT reliable. "
              f"Report this dataset descriptively (per unit), not as a test.")
        for u, g in zip(uniq, groups):
            print(f"   unit {u}: windows {len(g)}  mean Delta {g.mean():+.4f}")
    near = np.abs(F - tau) <= 0.05
    print("\n================ SUMMARY ================")
    print(f"dataset {a.dataset}  seed {a.seed}  arm {a.arm}  variant {a.variant}  windows {n}  units {n_clusters}  "
          f"queries {tot}  H={H}  Cm={Cm:g} Cf={Cf:g}  tau={tau:.3f}")
    used_t = {r.get("temperature", a.temperature) for r in recs}
    t_txt = "default (the model rejected the setting)" if used_t == {None} else (
        str(a.temperature) if used_t == {a.temperature} else f"mixed {sorted(map(str, used_t))}")
    print(f"prompt hash {prompt_hash(a.variant)}   model {a.model or a.backend}   temperature {t_txt}")
    print(f"parse rate               {parsed_ok / tot:.3f}   (unparseable are scored INTERVENE)")
    from collections import Counter
    unp = Counter(r.get("stop_reason", "unknown") for r in recs if r["decision"] is None)
    if unp:
        print(f"   unparseable answers by stop reason: {dict(unp)}   (max_tokens = answer was cut off: raise MAX_TOKENS)")
    print(f"critical windows (y<=H)  {int(fail.sum())} of {n}")
    print(f"mean cost  rule          {rule_c.mean():.4f}")
    print(f"mean cost  agent         {agent_c.mean():.4f}")
    print(f"mean cost  always-interv {Cm:.4f}   never-intervene {np.mean(np.where(fail, Cf, 0.0)):.4f}")
    print(f"Delta = agent - rule     {delta.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({100 * delta.mean() / max(rule_c.mean(), 1e-9):+.1f}% of rule cost)")
    print(f"agreement with the rule  {agree.mean():.3f}")
    print(f"repetition instability   {unstable.mean():.3f}   (share of windows whose repetitions differ)")
    print(f"windows near tau (|F-tau|<=0.05): {int(near.sum())}   far: {int((~near).sum())}")
    if near.any():
        print(f"   Delta near {delta[near].mean():+.4f}   agreement near {agree[near].mean():.3f}")
    if (~near).any():
        print(f"   Delta far  {delta[~near].mean():+.4f}   agreement far  {agree[~near].mean():.3f}")
    print("NOTE: a pilot of ~50 windows is only a format / sanity check, not a result.")
    return {"dataset": a.dataset, "seed": a.seed, "arm": a.arm, "variant": a.variant,
            "model": a.model or a.backend, "temperature": a.temperature, "Cm": Cm, "Cf": Cf,
            "H": H, "windows": int(n), "units": int(n_clusters), "queries": int(tot), "prompt_hash": prompt_hash(a.variant),
            "parse_rate": parsed_ok / tot, "unparsed_by_stop": dict(unp), "critical": int(fail.sum()),
            "cost_rule": float(rule_c.mean()), "cost_agent": float(agent_c.mean()),
            "delta": float(delta.mean()), "delta_lo": float(lo), "delta_hi": float(hi),
            "delta_pct_of_rule": float(100 * delta.mean() / max(rule_c.mean(), 1e-9)),
            "agreement": float(agree.mean()), "instability": float(unstable.mean()),
            "n_near": int(near.sum()), "n_far": int((~near).sum()),
            "delta_near": float(delta[near].mean()) if near.any() else None,
            "delta_far": float(delta[~near].mean()) if (~near).any() else None}


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="folder with <...>/<dataset>/seed*_preds.npz")
    p.add_argument("--dataset", default="FD001")
    p.add_argument("--seed", type=int, default=0, help="seed of the trained GRU")
    p.add_argument("--n", type=int, default=50, help="windows to query; 0 = all test engines")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--arm", choices=["F", "FR"], default="F")
    p.add_argument("--variant", choices=["main", "rule_stated", "noF"], default="main")
    p.add_argument("--Cm", type=float, default=1.0)
    p.add_argument("--Cf", type=float, default=10.0)
    p.add_argument("--knn_k", type=int, default=50)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--backend", choices=["openai", "anthropic", "mock"], default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--no_think", action="store_true",
                   help="openai backend only: ask the server to disable thinking mode (check your model's docs)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min_interval", type=float, default=0.0,
                   help="minimum seconds between two requests (all workers together); 1.2 for a limit of 1 request/second")
    p.add_argument("--out", default="agent_runs")
    p.add_argument("--dry_run", action="store_true", help="print 3 prompts and stop")
    p.add_argument("--summarize_only", action="store_true")
    a = p.parse_args()

    D = load_preds(a.results, a.dataset, a.seed)
    N = len(D["y"])
    print(f"loaded {D['file']}\n  {N} test windows, H={D['H']}")
    rng = np.random.RandomState(a.sample_seed)
    idx = np.arange(N) if a.n in (0, N) or a.n > N else np.sort(rng.choice(N, a.n, replace=False))

    if a.dry_run:
        for i in idx[:3]:
            print(f"\n----- window {i} (F={D['F'][i]:.4f}) -----\n{build_prompt(D, i, a.Cm, a.Cf, a.arm, a.variant, a.knn_k)}")
        print(f"\nprompt hash {prompt_hash(a.variant)}")
        return

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_s{a.seed}_{a.arm}_{a.variant}_{(a.model or a.backend).replace('/', '_').replace(':', '_')}_t{a.temperature}_Cm{a.Cm:g}_Cf{a.Cf:g}"
    path = os.path.join(a.out, tag + ".jsonl")
    done, recs = set(), []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("stop_reason") == "error" or r.get("response") is None:
                    continue  # a failed call is not an answer: it will be re-queried
                recs.append(r)
                done.add((r["idx"], r["rep"]))
        print(f"resuming: {len(done)} queries already in {path}")

    if not a.summarize_only:
        B = Backend(a)
        lock = threading.Lock()
        todo = [(int(i), r) for i in idx for r in range(a.reps) if (int(i), r) not in done]
        print(f"{len(todo)} queries to run -> {path}")

        def work(i, rep):
            pr = build_prompt(D, i, a.Cm, a.Cf, a.arm, a.variant, a.knn_k)
            txt, stop = B.ask(pr, hash((i, rep)) & 0xFFFFFFF)
            return {"idx": i, "rep": rep, "unit": int(D["unit"][i]), "F": float(D["F"][i]),
                    "y": float(D["y"][i]), "response": txt, "decision": parse_decision(txt),
                    "prompt_hash": prompt_hash(a.variant), "model": a.model or a.backend,
                    "temperature": a.temperature if B.temp_ok else None, "stop_reason": stop,
                    "max_tokens": a.max_tokens}

        with ThreadPoolExecutor(max_workers=a.workers) as ex, open(path, "a", encoding="utf-8") as f:
            futs = [ex.submit(work, i, r) for i, r in todo]
            for k, fu in enumerate(as_completed(futs), 1):
                r = fu.result()
                if r["stop_reason"] == "error":
                    n_fail = globals().get("_nf", 0) + 1
                    globals()["_nf"] = n_fail
                    continue  # not saved: re-run the same command to retry it
                with lock:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    recs.append(r)
                if k % 25 == 0 or k == len(futs):
                    print(f"  {k}/{len(futs)} done")
    if globals().get("_nf"):
        print(f"[WARNING] {globals()['_nf']} queries failed and were NOT saved; re-run the same command to retry them.")
    keep = set(int(i) for i in idx)
    recs = [r for r in recs if r["idx"] in keep]
    if not recs:
        sys.exit("no records to summarise")
    res = summarize(D, recs, a)
    with open(os.path.join(a.out, tag + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
