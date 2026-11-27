#!/usr/bin/env python3
"""
analyze_pooled.py - the primary analysis of paper 2, from the saved answers.

  python analyze_pooled.py agent_runs_cerebras_main
  python analyze_pooled.py agent_runs_cerebras_main --Cf 10 --model qwen-3.8-27b

Reads every  <DATASET>_s0_F_main_<model>_t*_Cm1_Cf<Cf>.jsonl  in the folder and prints
  1. one row per dataset (Delta with a 95% paired bootstrap over engines),
  2. the POOLED row over FD001-FD004 (primary analysis, one window per engine),
  3. the informative subset (F in [0.02, 0.8]) pooled over FD001-FD004,
  4. N-CMAPSS per unit (descriptive only: 4 units are too few for an interval).
No equivalence margin is used. Unparseable answers are scored INTERVENE (as in the protocol).
Writes analysis_pooled.md (and .json) in the same folder.
"""
import argparse, glob, json, os, sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--Cm", type=float, default=1.0)
p.add_argument("--Cf", type=float, default=10.0)
p.add_argument("--model", default="", help="part of the file name to select one model")
p.add_argument("--lo", type=float, default=0.02)
p.add_argument("--hi", type=float, default=0.8)
p.add_argument("--B", type=int, default=10000)
p.add_argument("--variant", default="main", help="main (F given) or noF (exploratory: F not given)")
a = p.parse_args()
tau = a.Cm / a.Cf
rng = np.random.RandomState(12345)


def cost(interv, fail):
    return np.where(interv, a.Cm, np.where(fail, a.Cf, 0.0))


def load(ds):
    pat = os.path.join(a.folder, f"{ds}_s0_F_{a.variant}_*{a.model}*_Cm{a.Cm:g}_Cf{a.Cf:g}.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    recs = [json.loads(l) for l in open(files[0], encoding="utf-8")]
    recs = [r for r in recs if r.get("stop_reason") != "error" and r.get("response") is not None]
    summ = files[0].replace(".jsonl", "_summary.json")
    H = json.load(open(summ))["H"] if os.path.exists(summ) else None
    return recs, H, files[0]


def per_window(recs, H):
    idxs = sorted({r["idx"] for r in recs})
    rows = []
    for i in idxs:
        rr = [r for r in recs if r["idx"] == i]
        F, y, u = rr[0]["F"], rr[0]["y"], rr[0]["unit"]
        fail = y <= H
        ds = np.array([r["decision"] != "WAIT" for r in rr])
        agent = float(np.mean(cost(ds, fail)))
        rule_i = F > tau
        rule = float(cost(np.array(rule_i), fail))
        rows.append((i, u, F, agent, rule, float(np.mean(ds == rule_i)), len(set(ds.tolist())) > 1,
                     float(np.mean([r["decision"] is not None for r in rr])),
                     float(np.mean(ds)) if not rule_i else 0.0,       # agent intervenes where the rule waits
                     float(np.mean(~ds)) if rule_i else 0.0))          # agent waits where the rule intervenes
    return np.array(rows, dtype=float)


def boot(delta, units):
    uniq = np.unique(units)
    groups = [delta[units == u] for u in uniq]
    out = np.empty(a.B)
    for b in range(a.B):
        pick = rng.randint(0, len(uniq), len(uniq))
        out[b] = np.concatenate([groups[j] for j in pick]).mean()
    return np.percentile(out, [2.5, 97.5])


lines, js = [], {}
def row(name, W, units):
    d = W[:, 3] - W[:, 4]
    lo, hi = boot(d, units)
    pct = 100 * d.mean() / max(W[:, 4].mean(), 1e-9)
    txt = (f"| {name} | {len(W)} | {W[:,7].mean():.3f} | {W[:,4].mean():.4f} | {W[:,3].mean():.4f} | "
           f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {pct:+.1f}% | {W[:,5].mean():.3f} | {W[:,6].mean():.3f} | "
           f"{W[:,8].mean():.3f} | {W[:,9].mean():.3f} |")
    lines.append(txt); print(txt)
    js[name] = dict(n=len(W), parse=W[:,7].mean(), rule=W[:,4].mean(), agent=W[:,3].mean(),
                    delta=d.mean(), lo=lo, hi=hi, agree=W[:,5].mean(), unstable=W[:,6].mean(),
                    over_intervene=W[:,8].mean(), under_intervene=W[:,9].mean())

head = ("| Set | Windows | Parse | Rule cost | Agent cost | Delta [95% CI] | % of rule | Agree | Unstable | Over-interv. | Under-interv. |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|")
print(f"variant = {a.variant}   tau = {tau:.3f}  (Cm={a.Cm:g}, Cf={a.Cf:g})\n"); print(head); lines.append(head)

pooled = []
for ds in ("FD001", "FD002", "FD003", "FD004"):
    L = load(ds)
    if not L:
        print(f"| {ds} | (no file found) |"); continue
    recs, H, f = L
    if H is None:
        print(f"[warn] {ds}: no summary json next to the answers (needed for H); skipping"); continue
    W = per_window(recs, H)
    row(ds, W, np.arange(len(W)))          # one window per engine: every window is its own unit
    pooled.append(W)

if len(pooled) == 4:
    P = np.vstack(pooled)
    row("POOLED C-MAPSS", P, np.arange(len(P)))
    m = (P[:, 2] >= a.lo) & (P[:, 2] <= a.hi)
    if m.any():
        row(f"Informative F in [{a.lo},{a.hi}]", P[m], np.arange(int(m.sum())))
    near = np.abs(P[:, 2] - tau) <= 0.05
    if near.any():
        row("Near tau (|F-tau|<=0.05)", P[near], np.arange(int(near.sum())))
else:
    print(f"\n[info] pooled row needs all four C-MAPSS files; found {len(pooled)} of 4.")

L = load("NCMAPSS")
if L:
    recs, H, f = L
    if H is not None:
        W = per_window(recs, H)
        lines.append("\nN-CMAPSS per unit (descriptive):\n\n| Unit | Windows | Rule | Agent | Delta |\n|---|---|---|---|---|")
        print("\nN-CMAPSS per unit (descriptive only)")
        for u in np.unique(W[:, 1]):
            M = W[W[:, 1] == u]
            t = f"| {int(u)} | {len(M)} | {M[:,4].mean():.4f} | {M[:,3].mean():.4f} | {(M[:,3]-M[:,4]).mean():+.4f} |"
            lines.append(t); print(t)

open(os.path.join(a.folder, "analysis_pooled.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(js, open(os.path.join(a.folder, "analysis_pooled.json"), "w"), indent=1)
print(f"\nwritten: {os.path.join(a.folder, 'analysis_pooled.md')}")
