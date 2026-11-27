#!/usr/bin/env python3
"""
run_all.py  -  one command that runs the whole paper-2 agent experiment.

  python run_all.py            # reads settings from .env (or from environment variables)
  python run_all.py --dry_run  # only prints what would run and how many LLM calls it needs
  python run_all.py --yes      # confirm a run larger than MAX_QUERIES
  (N_<DATASET> in .env limits one dataset, e.g. N_NCMAPSS=300)

It loops over  models x datasets x seeds x arms x prompt variants x Cf values,
calls pilot_agent.py for each combination (every call is resumable: finished queries are
never repeated), then collects every summary into  OUT_DIR/summary_all.csv  and  summary_all.md.

Settings: see .env.example.  On Kaggle, put the API key in Kaggle Secrets and export it
into the environment (see README_PACKAGE.md) instead of uploading a .env file.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path):
    """Minimal .env reader. The file wins for every setting, so an old value left in the terminal
    or in Windows cannot silently override it. Only secrets (names containing KEY) keep the rule
    'a real environment variable wins', so Kaggle Secrets keep working."""
    vals = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in vals.items():
        if "KEY" in k.upper():
            os.environ.setdefault(k, v)
        else:
            if os.environ.get(k, v) != v:
                print(f"[note] {k} was set outside .env ({os.environ[k]!r}); using the .env value {v!r}")
            os.environ[k] = v
    return vals


def env(name, default=None, cast=str):
    v = os.environ.get(name, "")
    if v == "":
        return default
    return cast(v)


def csv_list(name, default, cast=str):
    v = env(name)
    return [cast(x.strip()) for x in v.split(",") if x.strip()] if v else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.path.join(HERE, ".env"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="allow a run above MAX_QUERIES")
    a = ap.parse_args()
    print(f"settings read from {a.env} (exists: {os.path.exists(a.env)})")
    load_env(a.env)

    results = env("RESULTS_DIR")
    out = env("OUT_DIR", os.path.join(HERE, "agent_runs"))
    if not results:
        sys.exit("RESULTS_DIR is not set (folder that contains <dataset>/seed*_preds.npz). See .env.example")
    datasets = csv_list("DATASETS", ["FD001"])
    seeds = csv_list("SEEDS", [0], int)
    arms = csv_list("ARMS", ["F"])
    variants = csv_list("VARIANTS", ["main"])
    cfs = csv_list("CF_LIST", [10.0], float)
    models = csv_list("MODELS", ["mock:mock"])          # backend:model, e.g. anthropic:<id>, openai:<name>
    n = env("N", 50, int)
    reps = env("REPS", 3, int)
    temp = env("TEMPERATURE", 0.0, float)
    workers = env("WORKERS", 4, int)
    min_interval = env("MIN_INTERVAL", 0.0, float)
    max_tokens = env("MAX_TOKENS", 1000, int)
    no_think = env("NO_THINK", 0, int)
    base_url = env("OPENAI_BASE_URL", "http://localhost:8000/v1")
    max_queries = env("MAX_QUERIES", 2000, int)

    combos = [(m, d, s, arm, v, cf) for m in models for d in datasets for s in seeds
              for arm in arms for v in variants for cf in cfs]
    DEFAULT_SIZE = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
    def n_of(d):                                   # N_<DATASET> overrides N, e.g. N_NCMAPSS=300
        return env(f"N_{d}", n, int)
    def size_of(d):
        k = n_of(d)
        return k if k else DEFAULT_SIZE.get(d, 1000)   # N=0 means all windows (NCMAPSS size unknown: assume 1000)
    total = sum(size_of(d) * reps for (m, d, s_, arm, v, cf) in combos)
    per = total // max(len(combos), 1)
    print(f"{len(combos)} runs  x  ~{per} queries  =  ~{total} LLM calls "
          f"(models {models}, datasets {datasets}, seeds {seeds}, arms {arms}, variants {variants}, Cf {cfs})")
    for m in models:
        if m.split(":", 1)[0] == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is missing (put it in .env or in the environment).")
    if total > max_queries and not a.yes:
        sys.exit(f"Refusing to start: {total} calls is above MAX_QUERIES={max_queries}. "
                 f"Check the plan above, then add --yes (or raise MAX_QUERIES).")
    if a.dry_run:
        for c in combos[:10]:
            print("  would run:", c)
        return

    os.makedirs(out, exist_ok=True)
    for k, (m, d, s, arm, v, cf) in enumerate(combos, 1):
        backend, model = m.split(":", 1)
        cmd = [sys.executable, os.path.join(HERE, "pilot_agent.py"), "--results", results,
               "--dataset", d, "--seed", str(s), "--n", str(n_of(d)), "--arm", arm, "--variant", v,
               "--Cf", str(cf), "--reps", str(reps), "--backend", backend, "--model", model,
               "--temperature", str(temp), "--max_tokens", str(max_tokens),
               "--workers", str(workers), "--out", out, "--base_url", base_url,
               "--min_interval", str(min_interval)]
        if no_think and backend == "openai":
            cmd.append("--no_think")
        print(f"\n######## run {k}/{len(combos)}: {m} {d} seed{s} arm {arm} {v} Cf={cf:g}")
        r = subprocess.run(cmd)
        if r.returncode == 3:
            sys.exit("\nSTOPPED: the provider refused the call (daily limit reached, no credit, or bad key). "
                     "Finished answers are saved. Fix the cause or wait until tomorrow, then run the same command again.")
        if r.returncode != 0:
            print(f"[warn] run {k} failed with code {r.returncode}; continuing with the next one "
                  f"(re-run the command to resume it)")

    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(out, "*_summary.json")))]
    if not rows:
        sys.exit("no summaries found")
    keys = list(rows[0].keys())
    with open(os.path.join(out, "summary_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    show = ["model", "dataset", "seed", "arm", "variant", "Cf", "windows", "parse_rate", "cost_rule",
            "cost_agent", "delta", "delta_lo", "delta_hi", "agreement", "instability"]
    with open(os.path.join(out, "summary_all.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(show) + " |\n|" + "---|" * len(show) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in show) + " |\n")
    print(f"\nDone. {len(rows)} summaries -> {out}/summary_all.csv and summary_all.md")


if __name__ == "__main__":
    main()