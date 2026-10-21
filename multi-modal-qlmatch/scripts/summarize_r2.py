#!/usr/bin/env python
"""Round-2 comparison: every backbone under the R2P protocol + Pandora references -> tables (md/csv) and figure.

Usage: summarize_r2.py R2P_ROOT PANDORA_JSON OUT_PREFIX
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

KEY = {"evt_nu": ("auc", "event ν/cosmic AUC", True), "flavor": ("macro_f1", "flavor macro-F1", True),
       "energy": ("resolution", "E resolution (lower better)", False), "hit": ("f1_q", "hit charge-wtd F1", True)}
LABEL = {"qlfm_S_C": "qlfm-S C (MAE, 6.6M)", "qlfm_S_CL": "qlfm-S CL (MAE, 6.6M)", "qlfm_M_CL": "qlfm-M CL (MAE, 21M)",
         "qlfm_SD_CL": "qlfm-SD CL (self-distill., 21M)", "qlfm_SD_C": "qlfm-SD C (self-distill., 21M)",
         "fm4npp": "FM4NPP (Mamba2, 5M, per hit)", "panda": "Panda (PTv3+Sonata, 21M)",
         "polarmae": "PoLAr-MAE (ViT-S, 22M)"}
ORDER = ["qlfm_M_CL", "qlfm_S_CL", "qlfm_S_C", "qlfm_SD_CL", "qlfm_SD_C", "panda", "polarmae", "fm4npp"]


def load_r2p(root):
    d = defaultdict(list)          # (bb, task, mode, lf) -> [metric]
    for p in glob.glob(os.path.join(root, "*", "*", "*", "*", "metrics.json")):
        bb, task, mode = p.split(os.sep)[-5:-2]
        m = json.load(open(p))
        d[(bb, task, mode, m["label_frac"])].append(m["test"][KEY[task][0]])
    return d


def load_pandora(path):
    d = defaultdict(list)
    for r in json.load(open(path)):
        name = "Pandora direct" if r["method"] == "pandora_direct" else f"Pandora+BDT {r['input']}"
        d[(name, r["task"], r["label_frac"])].append(r["test"][KEY[r["task"]][0]])
    return d


def fmt(v):
    if not v:
        return "—"
    return f"{np.mean(v):.3f}" + (f"±{np.std(v):.3f}" if len(v) > 1 else "")


def main():
    root, pnd_path, out = sys.argv[1:4]
    r = load_r2p(root)
    p = load_pandora(pnd_path)
    lfs = (0.001, 0.01, 0.1)
    md = ["# Round-2 comparison (R2P protocol, test split)\n",
          "Frozen backbone + head at 0.1 / 1 / 10 % labels; fine-tune at 0.1 / 1 %; late-fused light for every "
          "backbone; 'C-only diag' = frozen at 1 % without late light.\n"]
    rows = []
    for task, (k, name, hib) in KEY.items():
        md.append(f"\n## {task}: {name}\n")
        md.append("| method | frozen 0.1% | frozen 1% | frozen 10% | FT 0.1% | FT 1% | C-only diag (frozen 1%) |")
        md.append("|---|---|---|---|---|---|---|")
        for bb in ORDER:
            cells = [fmt(r.get((bb, task, "frozen", lf))) for lf in lfs]
            cells += [fmt(r.get((bb, task, "finetune", lf))) for lf in lfs[:2]]
            cells += [fmt(r.get((bb, task, "frozen_nolight", 0.01)))]
            if all(c == "—" for c in cells):
                continue
            md.append(f"| {LABEL[bb]} | " + " | ".join(cells) + " |")
            rows.append(dict(task=task, method=bb, cells=cells))
        if task != "hit":
            for name_p in ("Pandora+BDT C", "Pandora+BDT CL"):
                cells = [fmt(p.get((name_p, task, lf))) for lf in lfs] + ["", "", ""]
                md.append(f"| {name_p} (no NN) | " + " | ".join(cells) + " |")
            md.append(f"| Pandora direct (no training) | {fmt(p.get(('Pandora direct', task, None)))} | | | | | |")
        else:
            md.append("| Pandora ν-slice hits (no training) | see round 1: 0.63–0.65 | | | | | |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.8))
    cmap = {"qlfm_M_CL": "#d62728", "qlfm_S_CL": "#ff7f0e", "qlfm_S_C": "#1f77b4", "qlfm_SD_CL": "#9467bd",
            "qlfm_SD_C": "#8c564b", "panda": "#2ca02c", "polarmae": "#17becf", "fm4npp": "#7f7f7f"}
    x = np.array(lfs) * 100
    for ax, (task, (k, name, hib)) in zip(axs, KEY.items()):
        for bb in ORDER:
            ys = [np.mean(r[(bb, task, "frozen", lf)]) if r.get((bb, task, "frozen", lf)) else np.nan for lf in lfs]
            if np.all(np.isnan(ys)):
                continue
            ax.plot(x, ys, marker="o", ms=7, lw=1.8, color=cmap[bb], label=LABEL[bb])
            yf = [np.mean(r[(bb, task, "finetune", lf)]) if r.get((bb, task, "finetune", lf)) else np.nan
                  for lf in lfs[:2]]
            ax.plot(x[:2], yf, marker="^", ms=8, lw=1.2, ls=":", color=cmap[bb])
        if task != "hit":
            yb = [np.mean(p[("Pandora+BDT CL", task, lf)]) for lf in lfs]
            ax.plot(x, yb, marker="s", ms=7, lw=2.2, ls="--", color="k", label="Pandora+BDT (+light)")
            yd = np.mean(p[("Pandora direct", task, None)])
            ax.axhline(yd, color="k", lw=1, ls="-.", label="Pandora direct")
        else:
            ax.axhline(0.64, color="k", lw=1, ls="-.", label="Pandora ν-slice hits")
        ax.set_xscale("log"); ax.set_xlabel("labeled fraction of train (%)", fontsize=11)
        ax.set_title(f"{task}: {name}", fontsize=11); ax.grid(alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.text(0.5, 0.005, "solid ●: frozen backbone + head;  dotted ▲: fine-tuned", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out + ".png", dpi=130)


if __name__ == "__main__":
    main()
