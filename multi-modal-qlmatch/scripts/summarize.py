#!/usr/bin/env python
"""Aggregate downstream metrics.json files into a CSV + markdown tables + label-efficiency plots.

Directory convention: ROOT/<task>/<mode>_<input>/f<frac>_s<seed>/metrics.json   (E1/E3)
                      ROOT/pre<pf>/<task>/<mode>/f<frac>_s<seed>/metrics.json     (E4)
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

KEY = {"evt_nu": ("auc", "ROC AUC"), "flavor": ("macro_f1", "macro-F1"),
       "energy": ("resolution", "E resolution (lower better)"), "hit": ("f1_q", "charge-wtd F1")}


def load(root):
    rows = []
    for p in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(p))
        rel = os.path.relpath(os.path.dirname(p), root).split(os.sep)
        r = {"path": p, "task": m["task"], "mode": m["mode"], "label_frac": m["label_frac"], "seed": m["seed"],
             "n_train": m["n_train"], "minutes": m["minutes"]}
        g = re.match(r"pre([0-9.]+)", rel[0])
        r["pre_frac"] = float(g.group(1)) if g else np.nan
        r["input"] = rel[-2].split("_")[-1] if not g else "CL"
        for k, v in m["test"].items():
            if np.isscalar(v):
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def table(df, task):
    k, name = KEY[task]
    d = df[(df.task == task) & df.pre_frac.isna()]
    if d.empty:
        return ""
    g = d.groupby(["mode", "input", "label_frac"])[k].agg(["mean", "std", "count"]).reset_index()
    g["val"] = g.apply(lambda r: f"{r['mean']:.3f}" + (f" ± {r['std']:.3f}" if r["count"] > 1 else ""), axis=1)
    piv = g.pivot_table(index=["mode", "input"], columns="label_frac", values="val", aggfunc="first")
    ntr = d.groupby("label_frac").n_train.median()
    piv.columns = [f"{c:g} (n={int(ntr[c])})" for c in piv.columns]
    return f"\n### {task}: {name}\n\n" + piv.to_markdown() + "\n"


def plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = [t for t in KEY if t in set(df.task)]
    fig, axs = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False)
    sty = {"scratch": "-", "frozen": "--", "finetune": ":", "linear": "-."}
    col = {"C": "#1f77b4", "CL": "#d62728"}
    for ax, t in zip(axs[0], tasks):
        k, name = KEY[t]
        d = df[(df.task == t) & df.pre_frac.isna()]
        for (mode, inp), g in d.groupby(["mode", "input"]):
            s = g.groupby("n_train")[k].agg(["mean", "std"]).reset_index()
            ax.errorbar(s.n_train, s["mean"], yerr=s["std"].fillna(0), ls=sty.get(mode, "-"), color=col.get(inp, "k"),
                        marker="o", ms=3, capsize=2, label=f"{mode} {inp}")
        ax.set_xscale("log")
        ax.set_xlabel("labeled training events")
        ax.set_title(f"{t}: {name}", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load(a.root)
    out = a.out or a.root
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    md = "".join(table(df, t) for t in KEY)
    open(os.path.join(out, "summary.md"), "w").write(md)
    print(md)
    if len(df):
        plot(df, os.path.join(out, "label_efficiency.png"))


if __name__ == "__main__":
    main()
