#!/usr/bin/env python
"""E4 (+E6): frozen-probe test metric and pretraining val loss vs pretraining-data fraction.

Usage: plot_scaling.py WORK_DIR OUT_PNG
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = {"evt_nu": ("auc", "event ν/cosmic AUC"), "flavor": ("macro_f1", "flavor macro-F1"),
     "energy": ("resolution", "E resolution (lower better)"), "hit": ("f1_q", "hit charge-wtd F1")}


def collect(W):
    pts = {}   # (model, inp, lf, task) -> {pre_frac: metric}
    for p in glob.glob(f"{W}/e4/*/*/frozen/*/metrics.json"):
        m = json.load(open(p))
        tag = p.split("/e4/")[1].split("/")[0]                 # e.g. cl_pre0.10
        inp, pf = tag.split("_pre")
        pts.setdefault(("S", inp, m["label_frac"], m["task"]), {})[float(pf)] = m["test"][K[m["task"]][0]]
    for inp in ("cl", "c"):                                     # 90% points from the main matrix
        for p in glob.glob(f"{W}/e3/*/frozen_{inp.upper()}/f*_s0/metrics.json"):
            m = json.load(open(p))
            if m["label_frac"] in (0.01, 0.1):
                pts.setdefault(("S", inp, m["label_frac"], m["task"]), {})[0.9] = m["test"][K[m["task"]][0]]
    for p in glob.glob(f"{W}/e6/*/*/frozen/*/metrics.json"):
        m = json.load(open(p))
        pf = float(p.split("/e6/")[1].split("/")[0].split("_pre")[1])
        pts.setdefault(("M", "cl", m["label_frac"], m["task"]), {})[pf] = m["test"][K[m["task"]][0]]
    vl = {}
    for d in glob.glob(f"{W}/pre/*"):
        name = os.path.basename(d)
        lines = [json.loads(l) for l in open(f"{d}/log.jsonl") if "val_loss" in l] if os.path.exists(f"{d}/log.jsonl") else []
        if not lines or not os.path.exists(f"{d}/done"):
            continue
        model = "M" if name.startswith("m_") else "S"
        inp, pf = name.replace("m_", "").split("_f")
        vl.setdefault((model, inp), {})[float(pf)] = lines[-1]["val_loss_c"]
    return pts, vl


def main():
    W, out = sys.argv[1], sys.argv[2]
    pts, vl = collect(W)
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.6))
    col = {("S", "cl"): "#d62728", ("S", "c"): "#1f77b4", ("M", "cl"): "#9467bd"}
    lab = {("S", "cl"): "CL (6.6M)", ("S", "c"): "C (6.6M)", ("M", "cl"): "CL (21M, 2x steps)"}
    for ax, t in zip(axs[:4], K):
        for (model, inp, lf, task), d in sorted(pts.items()):
            if task != t:
                continue
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], marker="s" if lf == 0.1 else "o", ms=7, lw=1.8, mew=1.6,
                    mfc=col[(model, inp)] if lf == 0.1 else "white", ls="-" if lf == 0.1 else "--",
                    color=col[(model, inp)], label=f"{lab[(model, inp)]}, {lf:g} labels")
        ax.set_xscale("log"); ax.set_xlabel("pretraining data fraction", fontsize=11)
        ax.set_title(K[t][1] + " (frozen)", fontsize=11)
        ax.grid(alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=11, handlelength=3.5, frameon=False)
    for key, d in sorted(vl.items()):
        xs = sorted(d)
        axs[4].plot(xs, [d[x] for x in xs], marker="D", ms=7, lw=1.8, color=col[key], label=lab[key])
    axs[4].set_xscale("log"); axs[4].set_xlabel("pretraining data fraction", fontsize=11)
    axs[4].legend(fontsize=10, handlelength=3)
    axs[4].set_title("pretraining val masked-charge loss (×mean)", fontsize=11); axs[4].grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=130)
    print(json.dumps({str(k): v for k, v in vl.items()}))


if __name__ == "__main__":
    main()
