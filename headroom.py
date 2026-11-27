"""
headroom.py  -  NO LLM, NO API, NO cost.  Runs on your saved CalibRAG outputs only.

What it answers (for every dataset, pooled C-MAPSS, and N-CMAPSS):
  1. ROUNDING CHECK   The prompt shows F with 4 decimals. Can that rounding flip the rule's
                      decision for any window? (the only way an agent could differ from the
                      rule by arithmetic alone)
  2. HEADROOM         Rule cost, always-intervene, never-intervene, and the ORACLE cost
                      (perfect knowledge of Y). Oracle - rule = the most any extra
                      information could ever save. Also: share of windows in the contested
                      zone near tau.
  3. Cf SWEEP         The same table for Cf = 5, 10, 20, 50 without any query.
  4. CALIBRATION      Reliability of F (bins of F vs observed failure rate, ECE).
  5. WHAT-IF          Cost of an agent that follows a DISTORTED F (x0.5, x2, +-0.1) or
                      decides without F (median only), computed offline. This is the
                      expected size of the effect the stress tests could show.
  6. EXACT BOUND      Clopper-Pearson 95% upper bound on the disagreement rate when 0 of n
                      windows disagree (replaces the rule of three).

Usage (from the inner paper2_agent folder, venv active):
  python headroom.py --results C:\\calibrag_results\\run_20261107_1613
  python headroom.py --results ... --datasets FD001,FD002,FD003,FD004,NCMAPSS --seed 0
Writes headroom.md and headroom.json in the current folder.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import norm, beta

Z80 = 1.2815515655446004
p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--datasets", default="FD001,FD002,FD003,FD004,NCMAPSS")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--Cm", type=float, default=1.0)
p.add_argument("--Cf", type=float, default=10.0)
p.add_argument("--cf_list", default="5,10,20,50")
p.add_argument("--near", type=float, default=0.05)
p.add_argument("--lo", type=float, default=0.02)
p.add_argument("--hi", type=float, default=0.8)
p.add_argument("--out", default="headroom")
a = p.parse_args()


def failure_prob(q10, q50, q90, H):
    sigma = np.maximum((q90 - q10) / (2 * Z80), 1e-6)
    return norm.cdf((H - q50) / sigma)


def load(ds):
    pat = os.path.join(a.results, "**", ds, f"seed{a.seed}_preds.npz")
    files = sorted(glob.glob(pat, recursive=True))
    if not files:
        print(f"[skip] {ds}: no file matches {pat}"); return None
    d = np.load(files[0], allow_pickle=True)
    H = int(d["H"])
    q10, q50, q90, y = (np.asarray(d[k], float) for k in ("q10C", "q50", "q90C", "y"))
    F = failure_prob(q10, q50, q90, H)
    unit = np.asarray(d["unit"]) if "unit" in d.files else np.arange(len(y))
    return dict(H=H, q10=q10, q50=q50, q90=q90, F=F, Y=(y <= H).astype(float), unit=unit, file=files[0])


def cost(interv, Y, Cm, Cf):
    return np.where(interv, Cm, Cf * Y)


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def ece(F, Y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(F, edges) - 1, 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append((float(edges[b]), float(edges[b + 1]), int(m.sum()), float(F[m].mean()), float(Y[m].mean())))
            e += m.sum() / len(F) * abs(F[m].mean() - Y[m].mean())
    return float(e), rows


def block(name, S, Cm, Cf):
    F, Y, q50, H = S["F"], S["Y"], S["q50"], S["H"]
    tau = Cm / Cf
    rule = F > tau
    c_rule = cost(rule, Y, Cm, Cf).mean()
    c_int = Cm
    c_wait = (Cf * Y).mean()
    c_orc = np.minimum(Cm, Cf * Y).mean()
    near = np.abs(F - tau) <= a.near
    inf = (F >= a.lo) & (F <= a.hi)
    Fr = np.round(F, 4)
    flips = int(((Fr > tau) != (F > tau)).sum())
    out = dict(set=name, n=int(len(F)), tau=tau, rule=float(c_rule), always_intervene=float(c_int),
               never_intervene=float(c_wait), oracle=float(c_orc), headroom=float(c_rule - c_orc),
               headroom_pct_of_rule=float(100 * (c_rule - c_orc) / c_rule) if c_rule > 0 else 0.0,
               informative=int(inf.sum()), near_tau=int(near.sum()), rounding_flips=flips,
               min_dist_to_tau=float(np.min(np.abs(F - tau))), cp_upper_0_of_n=cp_upper(0, len(F)),
               cp_upper_0_of_informative=cp_upper(0, max(int(inf.sum()), 1)))
    # what-if: agent follows a distorted F, or uses the median only
    wi = {}
    for lab, Fd in (("F x0.5", np.clip(F * 0.5, 0, 1)), ("F x2", np.clip(F * 2, 0, 1)),
                    ("F +0.1", np.clip(F + 0.1, 0, 1)), ("F -0.1", np.clip(F - 0.1, 0, 1))):
        d = Fd > tau
        wi[lab] = dict(cost=float(cost(d, Y, Cm, Cf).mean()), flips=int((d != rule).sum()))
    d = q50 < H                      # median-only agent: intervene if median life < horizon
    wi["median only (no F)"] = dict(cost=float(cost(d, Y, Cm, Cf).mean()), flips=int((d != rule).sum()))
    out["what_if"] = wi
    return out


sets, calib = {}, {}
for ds in [s.strip() for s in a.datasets.split(",") if s.strip()]:
    S = load(ds)
    if S is not None:
        sets[ds] = S

cm_ds = {k: v for k, v in sets.items() if k.startswith("FD")}
if len(cm_ds) > 1:
    keys = ("F", "Y", "q50")
    sets["POOLED C-MAPSS"] = dict(**{k: np.concatenate([v[k] for v in cm_ds.values()]) for k in keys},
                                  H=0, unit=None, file="pooled")
    # pooled what-if with per-dataset H for the median-only rule
    sets["POOLED C-MAPSS"]["H"] = None

results = {}
md = []
cfs = [float(x) for x in a.cf_list.split(",")]
for name, S in sets.items():
    if S["H"] is None:   # pooled: override median-only decision
        S2 = dict(S); S2["H"] = np.concatenate([np.full(len(v["F"]), v["H"]) for v in cm_ds.values()])
        S = S2
    results[name] = {f"Cf={cf:g}": block(name, S, a.Cm, cf) for cf in sorted(set(cfs + [a.Cf]))}
    e, rows = ece(S["F"], S["Y"])
    calib[name] = dict(ece=e, bins=rows)

main = f"Cf={a.Cf:g}"
md.append(f"# Headroom and rounding check (no LLM)  -  Cm={a.Cm:g}, Cf={a.Cf:g}, tau={a.Cm/a.Cf:.3f}, seed {a.seed}\n")
md.append("| Set | n | Rule | Always interv. | Never interv. | Oracle | Headroom (oracle-rule) | Informative | Near tau | Rounding flips | min abs(F-tau) | Exact 95% upper bound, 0 of n |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, r in results.items():
    x = r[main]
    md.append(f"| {name} | {x['n']} | {x['rule']:.4f} | {x['always_intervene']:.4f} | {x['never_intervene']:.4f} | {x['oracle']:.4f} | "
              f"{x['headroom']:.4f} ({x['headroom_pct_of_rule']:.0f}% of rule) | {x['informative']} | {x['near_tau']} | "
              f"{x['rounding_flips']} | {x['min_dist_to_tau']:.5f} | {100*x['cp_upper_0_of_n']:.2f}% |")
md.append("\nRounding flips = windows where F rounded to 4 decimals (as shown in the prompt) gives a different decision from the unrounded F.\n"
          "Headroom = what perfect knowledge of failure could save; no agent can save more than this, and a calibrated-F rule has none beyond it from F alone.\n")
md.append("\n## Cf sweep (rule cost / always / never / oracle)\n")
md.append("| Set | Cf | tau | Rule | Always | Never | Oracle | Informative | Near tau |")
md.append("|---|---|---|---|---|---|---|---|---|")
for name, r in results.items():
    for k, x in r.items():
        md.append(f"| {name} | {k[3:]} | {x['tau']:.3f} | {x['rule']:.4f} | {x['always_intervene']:.4f} | {x['never_intervene']:.4f} | {x['oracle']:.4f} | {x['informative']} | {x['near_tau']} |")
md.append("\n## What-if: cost of an agent that follows a distorted F, or ignores F (offline, no LLM)\n")
md.append("| Set | Variant | Cost | Rule cost | Cost - rule | Windows whose decision changes |")
md.append("|---|---|---|---|---|---|")
for name, r in results.items():
    x = r[main]
    for lab, w in x["what_if"].items():
        md.append(f"| {name} | {lab} | {w['cost']:.4f} | {x['rule']:.4f} | {w['cost']-x['rule']:+.4f} | {w['flips']} |")
md.append("\n## Calibration of F (10 bins)\n")
for name, c in calib.items():
    md.append(f"**{name}**: ECE = {c['ece']:.4f}\n")
    md.append("| F bin | n | mean F | observed failure rate |")
    md.append("|---|---|---|---|")
    for lo, hi, n, mf, my in c["bins"]:
        md.append(f"| {lo:.1f}-{hi:.1f} | {n} | {mf:.3f} | {my:.3f} |")
    md.append("")

open(a.out + ".md", "w", encoding="utf-8").write("\n".join(md))
json.dump(dict(results=results, calibration=calib), open(a.out + ".json", "w"), indent=1)
print("\n".join(md[:4 + len(results)]))
print(f"\nwritten: {a.out}.md  and  {a.out}.json")