"""
make_figures_paper2.py  -  figures for the IEEEtran paper.  No LLM, no API, no cost.

Reads (1) your saved CalibRAG outputs  seed*_preds.npz   and
      (2) the agent answers (.jsonl) of the two models.
Writes PDF (vector, for LaTeX) + PNG (preview) in ./figs_paper2

Usage (inner paper2_agent folder, venv active):
  python make_figures_paper2.py ^
     --results C:\\calibrag_results\\run_20261107_1613 ^
     --qwen_dir agent_runs_cerebras_main ^
     --gptoss_dir agent_runs_groq_gptoss_main

Figures
  fig1_decisions    agent and rule decisions along F: the switch happens exactly at tau
  fig2_cost         cost per window: oracle, rule, Qwen, gpt-oss, always-intervene
  fig3_hist         distribution of F with tau and the informative band
  fig4_reliability  calibration of F (reliability diagram, ECE)
  fig5_cf_sweep     how much the rule saves as Cf changes (offline, rule only)
  fig6_whatif       OFFLINE what-if of a distorted F / no F  (EXPLORATORY preview of the stress tests)
  fig7_nof_delta    EXPLORATORY: cost difference agent - rule with 95% CI, F given vs F not given (Qwen)
  fig8_nof_deviate  EXPLORATORY: where the no-F agent deviates from the rule, and agreement by distance to tau
Colors: blue / orange / aqua (validated colour-blind-safe set); every series also has a marker,
hatch or direct label, so the figures stay readable in grayscale print.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

p = argparse.ArgumentParser()
p.add_argument("--results", required=True)
p.add_argument("--qwen_dir", required=True)
p.add_argument("--gptoss_dir", required=True)
p.add_argument("--qwen_nof_dir", default="agent_runs_cerebras_noF", help="folder of the exploratory no-F run (Qwen); figures 7-8 are skipped if absent")
p.add_argument("--qwen_tag", default="qwen")
p.add_argument("--gptoss_tag", default="gpt-oss")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--Cm", type=float, default=1.0)
p.add_argument("--Cf", type=float, default=10.0)
p.add_argument("--lo", type=float, default=0.02)
p.add_argument("--hi", type=float, default=0.8)
p.add_argument("--out", default="figs_paper2")
p.add_argument("--with_whatif", action="store_true", help="also draw the offline what-if figure (fig6, exploratory; off by default)")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
TAU = a.Cm / a.Cf
Z80 = 1.2815515655446004
FD = ["FD001", "FD002", "FD003", "FD004"]

BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, MUTED, GRID, BAND = "#191919", "#666666", "#e6e6e6", "#efefef"
RULE_C, ORACLE_C, ALWAYS_C = "#3a3a3a", "#b0b0b0", "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "hatch.linewidth": 0.5,
})
COL1, COL2 = 3.5, 7.16   # IEEEtran column and page width (inches)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("written", os.path.join(a.out, name + ".pdf"))


def panel(ax, s):
    ax.text(0.0, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


# ------------------------------------------------------------------ data
def failure_prob(q10, q50, q90, H):
    return norm.cdf((H - q50) / np.maximum((q90 - q10) / (2 * Z80), 1e-6))


def load_npz(ds):
    files = sorted(glob.glob(os.path.join(a.results, "**", ds, f"seed{a.seed}_preds.npz"), recursive=True))
    if not files:
        print(f"[skip] no npz for {ds}"); return None
    d = np.load(files[0], allow_pickle=True)
    H = int(d["H"]); q10, q50, q90, y = (np.asarray(d[k], float) for k in ("q10C", "q50", "q90C", "y"))
    return dict(F=failure_prob(q10, q50, q90, H), Y=(y <= H).astype(float), q50=q50, H=np.full(len(y), H))


def pool(parts):
    parts = [x for x in parts if x is not None]
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]} if parts else None


def load_agent(folder, tag, ds, variant="main"):
    pat = os.path.join(folder, f"{ds}_s0_F_{variant}_*{tag}*_Cm{a.Cm:g}_Cf{a.Cf:g}.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"[skip] no agent file for {ds} / {tag} in {folder}"); return None
    recs = [json.loads(l) for l in open(files[0], encoding="utf-8")]
    recs = [r for r in recs if r.get("stop_reason") != "error" and r.get("response") is not None]
    sj = files[0].replace(".jsonl", "_summary.json")
    H = json.load(open(sj))["H"]
    by = {}
    for r in recs:
        by.setdefault(r["idx"], []).append(r)
    F, Y, D = [], [], []
    for i in sorted(by):
        rr = by[i]
        F.append(rr[0]["F"]); Y.append(float(rr[0]["y"] <= H))
        D.append(np.mean([r["decision"] != "WAIT" for r in rr]) >= 0.5)   # unparseable counts as INTERVENE
    return dict(F=np.array(F), Y=np.array(Y), D=np.array(D, bool))


def pool_agent(folder, tag, dss, variant="main"):
    parts = [load_agent(folder, tag, d, variant) for d in dss]
    parts = [x for x in parts if x is not None]
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]} if parts else None


def cost(dec, Y, Cf=None):
    return np.where(dec, a.Cm, (a.Cf if Cf is None else Cf) * Y)


# sets: name -> (qwen, gptoss) agent data
SETS = {}
for ds in FD:
    SETS[ds] = (load_agent(a.qwen_dir, a.qwen_tag, ds), load_agent(a.gptoss_dir, a.gptoss_tag, ds))
SETS["Pooled\nC-MAPSS"] = (pool_agent(a.qwen_dir, a.qwen_tag, FD), pool_agent(a.gptoss_dir, a.gptoss_tag, FD))
SETS["N-CMAPSS\n(300)"] = (load_agent(a.qwen_dir, a.qwen_tag, "NCMAPSS"), load_agent(a.gptoss_dir, a.gptoss_tag, "NCMAPSS"))
SETS = {k: v for k, v in SETS.items() if v[0] is not None and v[1] is not None}

NPZ = {ds: load_npz(ds) for ds in FD + ["NCMAPSS"]}
NPZ["POOLED"] = pool([NPZ[d] for d in FD])
NPZ = {k: v for k, v in NPZ.items() if v is not None}


# ------------------------------------------------------------------ fig 1: decisions along F
def fig1():
    names = [n for n in SETS if n.startswith("Pooled") or n.startswith("N-C")]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(COL2, 2.15), squeeze=False)
    rng = np.random.RandomState(1)
    for ax, nm, lab in zip(axes[0], names, "ab"):
        q, g = SETS[nm]
        F = q["F"]
        xlo = 1e-4 if nm.startswith("Pooled") else 1e-2      # N-CMAPSS never has F below ~0.04
        rows = [("Rule", F > TAU), ("Qwen 3.8 27B", q["D"]), ("gpt-oss-120b", g["D"])]
        ax.axvspan(a.lo, a.hi, color=BAND, zorder=0, lw=0)
        for r, (lbl, dec) in enumerate(rows[::-1]):
            y = r + rng.uniform(-0.22, 0.22, len(F))
            x = np.clip(F, xlo, 1.0)
            w = ~dec
            ax.scatter(x[w], y[w], s=5, marker="o", color=BLUE, alpha=0.55, lw=0, zorder=3)
            ax.scatter(x[dec], y[dec], s=6, marker="^", color=ORANGE, alpha=0.65, lw=0, zorder=3)
        ax.axvline(TAU, color=INK, lw=0.9, ls="--", zorder=4)
        ax.text(TAU * 1.12, 2.62, r"$\tau=%.2f$" % TAU, fontsize=7, va="top")
        ax.set_xscale("log"); ax.set_xlim(xlo, 1.0); ax.set_ylim(-0.6, 2.7)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel(r"calibrated failure probability $F$ (log scale; values below $10^{-4}$ shown at $10^{-4}$)" if lab == "a" else r"calibrated failure probability $F$ (log scale)")
        ax.grid(axis="y", visible=False)
        dis = int(((F > TAU) != q["D"]).sum() + ((F > TAU) != g["D"]).sum())
        ax.set_title(f"{nm.replace(chr(10), ' ')}: {len(F)} windows, {dis} disagreements with the rule", loc="left", fontsize=7.5)
    h = [Line2D([], [], marker="^", ls="", color=ORANGE, ms=4, label="INTERVENE"),
         Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="WAIT"),
         Patch(facecolor=BAND, label=f"informative zone $F\\in[{a.lo:g},{a.hi:g}]$")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_decisions")


# ------------------------------------------------------------------ fig 2: cost per window
def fig2():
    names = list(SETS)
    pol = [("Oracle", ORACLE_C, None), ("Rule", RULE_C, None), ("Qwen 3.8 27B", BLUE, "////"),
           ("gpt-oss-120b", AQUA, "\\\\\\\\"), ("Always intervene", ALWAYS_C, None)]
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    w = 0.15
    for j, (lab, col, hat) in enumerate(pol):
        vals = []
        for nm in names:
            q, g = SETS[nm]
            Y, F = q["Y"], q["F"]
            vals.append({"Oracle": np.minimum(a.Cm, a.Cf * Y).mean(),
                         "Rule": cost(F > TAU, Y).mean(),
                         "Qwen 3.8 27B": cost(q["D"], Y).mean(),
                         "gpt-oss-120b": cost(g["D"], g["Y"]).mean(),
                         "Always intervene": a.Cm}[lab])
        ax.bar(np.arange(len(names)) + (j - 2) * w, vals, w * 0.9, color=col, hatch=hat,
               edgecolor="white" if hat is None else INK, lw=0.4, label=lab, zorder=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel(r"mean cost per window ($C_m=1$, $C_f=10$)"); ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "fig2_cost")


# ------------------------------------------------------------------ fig 3: distribution of F
def fig3():
    keys = [k for k in ("POOLED", "NCMAPSS") if k in NPZ]
    titles = {"POOLED": "Pooled C-MAPSS", "NCMAPSS": "N-CMAPSS (all test windows)"}
    fig, axes = plt.subplots(1, len(keys), figsize=(COL2, 2.0), squeeze=False)
    for ax, k, lab in zip(axes[0], keys, "ab"):
        F = NPZ[k]["F"]
        ax.axvspan(a.lo, a.hi, color=BAND, zorder=0, lw=0)
        ax.hist(F, bins=np.linspace(0, 1, 26), color=BLUE, edgecolor="white", lw=0.4, zorder=3)
        ax.axvline(TAU, color=INK, lw=0.9, ls="--", zorder=4)
        ax.set_yscale("log"); ax.set_xlim(0, 1)
        n_inf = int(((F >= a.lo) & (F <= a.hi)).sum())
        n_near = int((np.abs(F - TAU) <= 0.05).sum())
        ax.set_title(f"{titles[k]}: {len(F)} windows; {n_inf} informative ({100*n_inf/len(F):.0f}%), {n_near} near $\\tau$", loc="left", fontsize=7.2)
        ax.set_xlabel(r"calibrated failure probability $F$"); ax.set_ylabel("windows (log scale)" if lab == "a" else "")
        ax.grid(axis="x", visible=False)
    h = [Patch(facecolor=BAND, label=f"informative zone $F\\in[{a.lo:g},{a.hi:g}]$"),
         Line2D([], [], color=INK, ls="--", lw=0.9, label=r"threshold $\tau=C_m/C_f$")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_hist")


# ------------------------------------------------------------------ fig 4: reliability
def reliability(F, Y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(F, edges) - 1, 0, bins - 1)
    pts, e = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            pts.append((F[m].mean(), Y[m].mean(), int(m.sum())))
            e += m.sum() / len(F) * abs(F[m].mean() - Y[m].mean())
    return np.array(pts), e


def fig4():
    fig, ax = plt.subplots(figsize=(COL1, 3.1))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", zorder=2)
    ax.axvline(TAU, color=INK, lw=0.8, ls="--", zorder=2)
    ax.text(TAU + 0.01, 0.98, r"$\tau$", fontsize=7, va="top")
    for k, lab, col, mk in (("POOLED", "Pooled C-MAPSS", BLUE, "o"), ("NCMAPSS", "N-CMAPSS", ORANGE, "s")):
        if k not in NPZ:
            continue
        pts, e = reliability(NPZ[k]["F"], NPZ[k]["Y"])
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.0, zorder=3)
        n_, p_ = pts[:, 2], pts[:, 1]                      # Wilson 95% interval of the observed frequency
        z = 1.96; cen = (p_ + z * z / (2 * n_)) / (1 + z * z / n_)
        half = z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / (1 + z * z / n_)
        ax.errorbar(pts[:, 0], cen, yerr=[cen - (cen - half), (cen + half) - cen], fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5, alpha=0.7, zorder=3)
        ax.scatter(pts[:, 0], pts[:, 1], s=6 + 90 * np.sqrt(pts[:, 2] / pts[:, 2].max()), color=col, marker=mk,
                   edgecolor="white", lw=0.5, zorder=4, label=f"{lab} (ECE {e:.3f})")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("mean predicted failure probability $F$ (10 bins)"); ax.set_ylabel("observed failure frequency")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, title="marker area $\\propto$ windows in bin; bars: Wilson 95% interval", title_fontsize=6.0)
    save(fig, "fig4_reliability")


# ------------------------------------------------------------------ fig 5: Cf sweep (rule only)
def fig5():
    cfs = np.array([2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100], float)
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for k, lab, col, mk in (("POOLED", "Pooled C-MAPSS", BLUE, "o"), ("NCMAPSS", "N-CMAPSS", ORANGE, "s")):
        if k not in NPZ:
            continue
        F, Y = NPZ[k]["F"], NPZ[k]["Y"]
        rule, orc = [], []
        for cf in cfs:
            tau = a.Cm / cf
            triv = min(a.Cm, (cf * Y).mean())
            rule.append(cost(F > tau, Y, cf).mean() / triv)
            orc.append(np.minimum(a.Cm, cf * Y).mean() / triv)
        ax.plot(cfs, rule, color=col, marker=mk, ms=3.2, lw=1.1, label=f"{lab}: rule")
        ax.plot(cfs, orc, color=col, lw=1.0, ls="--", label=f"{lab}: oracle")
    ax.axvline(a.Cf, color=INK, lw=0.8, ls=":", label=f"main setting $C_f={a.Cf:g}$")
    ax.axhline(1.0, color=MUTED, lw=0.7, label="best fixed policy (always or never intervene)")
    ax.set_xscale("log"); ax.set_ylim(0, 1.12); ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel(r"failure cost $C_f$ ($C_m=1$)"); ax.set_ylabel("cost / cost of best fixed policy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.2, columnspacing=1.0)
    save(fig, "fig5_cf_sweep")


# ------------------------------------------------------------------ fig 6: offline what-if (EXPLORATORY)
def fig6():
    keys = [k for k in ("POOLED", "NCMAPSS") if k in NPZ]
    vars_ = [("$F\\times0.5$", lambda S: np.clip(S["F"] * 0.5, 0, 1) > TAU), ("$F\\times2$", lambda S: np.clip(S["F"] * 2, 0, 1) > TAU),
             ("$F\\times0.25$", lambda S: np.clip(S["F"] * 0.25, 0, 1) > TAU), ("$F\\times4$", lambda S: np.clip(S["F"] * 4, 0, 1) > TAU),
             ("median only\n(no $F$)", lambda S: S["q50"] < S["H"])]
    fig, ax = plt.subplots(figsize=(COL1 + 0.6, 2.6))
    w = 0.36
    for j, (k, lab, col, hat) in enumerate((("POOLED", "Pooled C-MAPSS", BLUE, "////"), ("NCMAPSS", "N-CMAPSS", ORANGE, "\\\\\\\\"))):
        if k not in NPZ:
            continue
        S = NPZ[k]; base = cost(S["F"] > TAU, S["Y"]).mean()
        vals = [cost(f(S), S["Y"]).mean() - base for _, f in vars_]
        ax.bar(np.arange(len(vars_)) + (j - 0.5) * w, vals, w * 0.92, color=col, hatch=hat, edgecolor=INK, lw=0.4, label=lab, zorder=3)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(vars_))); ax.set_xticklabels([v[0] for v in vars_], fontsize=6.5)
    ax.set_ylabel("extra cost per window vs. the rule"); ax.grid(axis="x", visible=False)
    ax.set_title("Offline what-if, exploratory (no LLM)", loc="left", fontsize=7.5)
    ax.legend(loc="upper left")
    save(fig, "fig6_whatif")


# ------------------------------------------------------------------ figs 7-8: EXPLORATORY no-F run
def nof_sets():
    if not os.path.isdir(a.qwen_nof_dir):
        print(f"[skip] figures 7-8: folder {a.qwen_nof_dir} not found"); return None
    out = {}
    for nm, dss in (("FD001", ["FD001"]), ("FD002", ["FD002"]), ("FD003", ["FD003"]), ("FD004", ["FD004"]),
                    ("Pooled\nC-MAPSS", FD), ("N-CMAPSS\n(300)", ["NCMAPSS"])):
        m = pool_agent(a.qwen_dir, a.qwen_tag, dss, "main")
        n = pool_agent(a.qwen_nof_dir, a.qwen_tag, dss, "noF")
        if m is not None and n is not None and len(m["F"]) == len(n["F"]):
            out[nm] = (m, n)
        elif m is not None and n is not None:
            print(f"[skip] {nm.replace(chr(10), ' ')}: main run has {len(m['F'])} windows, no-F run {len(n['F'])} (same N needed)")
    return out or None


def boot_ci(delta, B=10000, seed=12345):
    if len(delta) == 0 or np.all(delta == delta[0]):
        return float(delta.mean()) if len(delta) else 0.0, float(delta.mean()) if len(delta) else 0.0
    rng = np.random.RandomState(seed)
    m = delta[rng.randint(0, len(delta), (B, len(delta)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def fig7():
    S = nof_sets()
    if S is None:
        return
    rows = []     # (label, delta_main, ci_main, delta_nof, ci_nof, n)
    def add(label, m, n, mask, ci=True):
        R = m["F"][mask] > TAU
        dm = cost(m["D"][mask], m["Y"][mask]) - cost(R, m["Y"][mask])
        dn = cost(n["D"][mask], n["Y"][mask]) - cost(R, n["Y"][mask])
        cm_, cn_ = (boot_ci(dm), boot_ci(dn)) if ci else ((dm.mean(),) * 2, (dn.mean(),) * 2)   # 4 units: no valid interval
        rows.append((label, dm.mean(), cm_, dn.mean(), cn_, int(mask.sum())))
    for nm, (m, n) in S.items():
        if not nm.startswith("N-C"):
            add(nm.replace("\n", " "), m, n, np.ones(len(m["F"]), bool))
    if "Pooled\nC-MAPSS" in S:
        m, n = S["Pooled\nC-MAPSS"]
        add("Informative (pooled)", m, n, (m["F"] >= a.lo) & (m["F"] <= a.hi))
        add("Near $\\tau$ (pooled)", m, n, np.abs(m["F"] - TAU) <= 0.05)
    for nm, (m, n) in S.items():
        if nm.startswith("N-C"):
            add("N-CMAPSS, 4 units: point only", m, n, np.ones(len(m["F"]), bool), ci=False)
    fig, ax = plt.subplots(figsize=(COL1 + 0.9, 3.3))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, dm, cm, dn, cn, n) in zip(y, rows):
        ax.plot([cm[0], cm[1]], [yy + 0.13] * 2, color=BLUE, lw=1.4, zorder=3)
        ax.plot(dm, yy + 0.13, "o", color=BLUE, ms=4, zorder=4)
        ax.plot([cn[0], cn[1]], [yy - 0.13] * 2, color=ORANGE, lw=1.4, zorder=3)
        ax.plot(dn, yy - 0.13, "s", color=ORANGE, ms=4, zorder=4)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{r[0]} ({r[5]})" for r in rows]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("cost difference per window, agent $-$ rule  (95% bootstrap interval)")
    h = [Line2D([], [], color=BLUE, marker="o", lw=1.4, ms=4, label=r"$F$ given (main)"),
         Line2D([], [], color=ORANGE, marker="s", lw=1.4, ms=4, label=r"$F$ not given (exploratory)")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig7_nof_delta")


def fig8():
    S = nof_sets()
    if S is None:
        return
    pan = [("Pooled\nC-MAPSS", (1e-4, 1.0)), ("N-CMAPSS\n(300)", (1e-2, 1.0))]
    pan = [p_ for p_ in pan if p_[0] in S]
    fig = plt.figure(figsize=(COL2, 2.4))
    gs = fig.add_gridspec(1, len(pan) + 1, width_ratios=[3] * len(pan) + [2.7], wspace=0.4)
    rng = np.random.RandomState(2)
    for k, (nm, xl) in enumerate(pan):
        ax = fig.add_subplot(gs[0, k]); n = S[nm][1]
        F, D = n["F"], n["D"]; R = F > TAU; dis = D != R
        yy = np.where(D, 1.0, 0.0) + rng.uniform(-0.2, 0.2, len(F)); x = np.clip(F, xl[0], 1.0)
        ax.scatter(x[~dis], yy[~dis], s=4, color="#9a9a9a", alpha=0.55, lw=0, zorder=3)
        ax.scatter(x[dis], yy[dis], s=26, marker="X", color=ORANGE, edgecolor=INK, lw=0.5, zorder=5)
        ax.axvline(TAU, color=INK, lw=0.9, ls="--"); ax.set_xscale("log"); ax.set_xlim(*xl); ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["WAIT", "INTERVENE"]); ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"$F$ (hidden from the agent)")
        ax.set_title(f"{nm.split(chr(10))[-1] if nm.startswith('Pooled') else 'N-CMAPSS'}: {int(dis.sum())}/{len(F)} differ", loc="left", fontsize=7.2)
        if k > 0:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[0, len(pan)])
    bands = [("near $\\tau$\n($\\leq$0.05)", lambda F: np.abs(F - TAU) <= 0.05),
             ("informative,\nfar", lambda F: (F >= a.lo) & (F <= a.hi) & (np.abs(F - TAU) > 0.05)),
             ("outside\ninformative", lambda F: (F < a.lo) | (F > a.hi))]
    w = 0.38
    for j, (nm, col, hat) in enumerate((("Pooled\nC-MAPSS", BLUE, "////"), ("N-CMAPSS\n(300)", ORANGE, "\\\\\\\\"))):
        if nm not in S:
            continue
        n = S[nm][1]; F, D = n["F"], n["D"]; R = F > TAU
        for b, (bl, fn) in enumerate(bands):
            m = fn(F); k = int((D[m] != R[m]).sum()); tot = int(m.sum())
            v = 1 - k / tot if tot else np.nan
            ax.bar(b + (j - 0.5) * w, v, w * 0.92, color=col, hatch=hat, edgecolor=INK, lw=0.4, zorder=3,
                   label=nm.replace("\n", " ") if b == 0 else None)
            ax.text(b + (j - 0.5) * w, min(v, 1) + 0.012, f"{k}/{tot}", ha="center", va="bottom", rotation=90, fontsize=5.4, color=MUTED)
    ax.set_ylim(0.5, 1.28); ax.set_yticks([0.6, 0.8, 1.0]); ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bands], fontsize=5.4)
    ax.set_ylabel("agreement with the rule"); ax.grid(axis="x", visible=False); ax.legend(loc="upper center", ncol=2, fontsize=5.8, columnspacing=0.8, handlelength=1.2)
    ax.set_title("by distance to $\\tau$", loc="left", fontsize=7.2)
    save(fig, "fig8_nof_deviate")


for f in (fig1, fig2, fig3, fig4, fig5, fig7, fig8) + ((fig6,) if a.with_whatif else ()):
    try:
        f()
    except Exception as e:     # one failing figure must not stop the others
        print(f"[error] {f.__name__}: {type(e).__name__}: {e}")
print("done ->", os.path.abspath(a.out))
