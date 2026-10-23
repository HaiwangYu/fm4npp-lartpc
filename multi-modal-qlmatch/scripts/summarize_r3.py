#!/usr/bin/env python
"""Round 3 (fairness controls, issue #5): LaTeX table fragments + figure from the R2P (round 2) and round-3 runs.
Runs that are planned but not finished yet are printed as \\pend, so the tech note can be rebuilt as results arrive.

Usage: summarize_r3.py R2P_ROOT R3_ROOT OUT_DIR(tex fragments) FIG_PNG
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["evt_nu", "flavor", "energy", "hit"]
TNAME = {"evt_nu": "T2 AUC", "flavor": "T3 F1", "energy": "T4 res.", "hit": "T1 F1$_q$"}
KEY = {"evt_nu": "auc", "flavor": "macro_f1", "energy": "resolution", "hit": "f1_q"}
HIGHER = {"evt_nu": True, "flavor": True, "energy": False, "hit": True}
# (label, root, key, has_xf)
ROWS = [("qlfm-M CL", "r2p", "qlfm_M_CL", True), ("qlfm-M C", "r3", "qlfm_M_C", True),
        ("qlfm-S CL", "r2p", "qlfm_S_CL", True), ("qlfm-S C", "r2p", "qlfm_S_C", True),
        ("Panda", "r2p", "panda", True), ("PoLAr-MAE", "r2p", "polarmae", True),
        ("FM4NPP m3, capped", "r2p", "fm4npp", False), ("FM4NPP m3", "r3", "fm4npp", True),
        ("FM4NPP m4", "r3", "fm4npp_m4", True)]
QLFM = ("qlfm_M_CL", "qlfm_S_CL", "qlfm_S_C", "qlfm_M_C")


def vals(root, key, task, mode, lf):
    out = []
    for p in sorted(glob.glob(f"{root}/{key}/{task}/{mode}/f{lf}_s*/metrics.json")):
        out.append(json.load(open(p))["test"][KEY[task]])
    return out


def cell(v, best=False, planned=True):
    if not v:
        return r"\pend" if planned else "--"
    s = f"{np.mean(v):.3f}"
    if len(v) > 1:
        s += f"$\\pm${np.std(v):.3f}".replace("0.", ".", 1)
    return f"\\textbf{{{s}}}" if best else s


def bold_mask(columns, task):
    """columns: list of value lists -> which entries are best (by mean)"""
    m = [np.mean(v) if v else np.nan for v in columns]
    if all(np.isnan(m)):
        return [False] * len(m)
    b = np.nanmax(m) if HIGHER[task] else np.nanmin(m)
    return [not np.isnan(x) and abs(x - b) < 5e-4 for x in m]


def main():
    r2p, r3, out, fig = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    root = {"r2p": r2p, "r3": r3}
    n_done = n_plan = 0

    # ---- (1) late light (R2P) vs strong fusion (XF), frozen, per label fraction ----
    for lf in ("0.01", "0.1"):
        data = {}
        for lab, rt, key, xf in ROWS:
            for t in TASKS:
                data[(key, rt, t, "ll")] = vals(root[rt], key, t, "frozen", lf)
                data[(key, rt, t, "xf")] = vals(r3, key, t, "frozen_xf", lf) if xf else []
        lines = []
        bests = {}
        for t in TASKS:
            col = [data[(k, rt, t, m)] for _, rt, k, _ in ROWS for m in ("ll", "xf")]
            bests[t] = bold_mask(col, t)
        for i, (lab, rt, key, xf) in enumerate(ROWS):
            cells = []
            for t in TASKS:
                for j, m in enumerate(("ll", "xf")):
                    v = data[(key, rt, t, m)]
                    planned = xf or m == "ll"
                    n_plan += planned
                    n_done += bool(v)
                    cells.append(cell(v, bests[t][2 * i + j], planned))
            lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        open(f"{out}/fusion_{lf}.tex", "w").write("\n".join(lines) + "\n")

    # ---- (2) full R2P matrix per task ----
    cols = [("frozen", "0.001"), ("frozen", "0.01"), ("frozen", "0.1"), ("finetune", "0.001"), ("finetune", "0.01"),
            ("frozen_nolight", "0.01")]
    mrows = [r for r in ROWS if r[2] in ("qlfm_M_CL", "qlfm_M_C", "qlfm_S_C", "panda", "fm4npp", "fm4npp_m4")]
    for t in TASKS:
        tab = [[vals(root[rt], key, t, m, lf) for m, lf in cols] for _, rt, key, _ in mrows]
        bests = [bold_mask([tab[i][c] for i in range(len(mrows))], t) for c in range(len(cols))]
        lines = []
        for i, (lab, rt, key, _) in enumerate(mrows):
            n_plan += len(cols)
            n_done += sum(bool(v) for v in tab[i])
            lines.append(f"{lab} & " + " & ".join(cell(tab[i][c], bests[c][i]) for c in range(len(cols))) + r" \\")
        open(f"{out}/matrix_{t}.tex", "w").write("\n".join(lines) + "\n")

    # ---- (3) T1 hit features (frozen 1 %) ----
    hrows = [r for r in ROWS if r[2] != "fm4npp" or r[1] == "r3"]
    tab = []
    for lab, rt, key, _ in hrows:
        native = vals(root[rt], key, "hit", "frozen", "0.01")
        raw = vals(r3, key, "hit", "frozen_raw", "0.01")
        none = vals(r3, key, "hit", "frozen_nohit", "0.01") if key in QLFM else None
        tab.append((lab, key, native, raw, none, vals(r3, key, "hit", "frozen_xf", "0.01")))
    lines = []
    for lab, key, native, raw, none, xf in tab:
        n_plan += 3 + (none is not None)
        n_done += bool(native) + bool(raw) + bool(xf) + bool(none)
        nat = cell(native) + ("" if key in QLFM else r"$^\ast$")
        lines.append(f"{lab} & {nat} & {cell(raw)} & {cell(none, planned=none is not None)} & {cell(xf)}" + r" \\")
    open(f"{out}/hitfeat.tex", "w").write("\n".join(lines) + "\n")
    open(f"{out}/status.tex", "w").write(f"\\newcommand{{\\rdone}}{{{n_done}}}\\newcommand{{\\rplan}}{{{n_plan}}}\n")

    # ---- figure: late light vs strong fusion, frozen 1 % and 10 % ----
    frows = [r for r in ROWS if r[3]]
    fig_, axs = plt.subplots(2, 4, figsize=(15, 6.2), sharey=False)
    x = np.arange(len(frows))
    for a, lf in enumerate(("0.01", "0.1")):
        for b, t in enumerate(TASKS):
            ax = axs[a, b]
            ll = [np.mean(v) if (v := vals(root[rt], k, t, "frozen", lf)) else np.nan for _, rt, k, _ in frows]
            xf = [np.mean(v) if (v := vals(r3, k, t, "frozen_xf", lf)) else np.nan for _, rt, k, _ in frows]
            ax.bar(x - 0.2, ll, 0.4, label="late light (R2P)", color="tab:gray")
            ax.bar(x + 0.2, xf, 0.4, label="strong late fusion", color="tab:orange")
            for i, (u, v) in enumerate(zip(ll, xf)):
                if np.isnan(v):
                    ax.text(i + 0.2, np.nanmin(ll) if not np.all(np.isnan(ll)) else 0, "pend.", rotation=90,
                            ha="center", va="bottom", fontsize=6, color="tab:red")
            lo = np.nanmin(ll + xf)
            hi = np.nanmax(ll + xf)
            ax.set_ylim(max(0, lo - 0.15 * (hi - lo + 1e-3)), hi + 0.1 * (hi - lo + 1e-3))
            ax.set_xticks(x)
            ax.set_xticklabels([r[0] for r in frows], rotation=60, ha="right", fontsize=7)
            ax.set_title(f"{TNAME[t].replace('$_q$', '_q')}{'' if HIGHER[t] else ' (lower better)'}, "
                         f"{float(lf):.0%} labels", fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.3)
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig_.tight_layout()
    fig_.savefig(fig, dpi=170)
    print(f"done {n_done}/{n_plan} table cells")


if __name__ == "__main__":
    main()
