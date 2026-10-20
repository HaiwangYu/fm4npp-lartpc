#!/usr/bin/env python
"""Agreement statistics behind the round-2 easy/hard event selection (same correctness rules as r2_select_events.py).

Writes OUT.json (per task: histogram of #algorithms correct, per-algorithm accuracy overall, on the consensus-hard
events and on the boundary events (about half of the algorithms correct),
dominant (truth -> majority prediction) patterns among them) and OUT.png (histograms).
Usage: r2_agreement.py OUT_PREFIX
"""
import json
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from r2_select_events import W, event_level, hit_level

TITLE = {"hit": "T1 hit (event F1$_q$ ≥ 0.8)", "evt_nu": "T2 event ν (class)", "flavor": "T3 flavor (class)",
         "energy": "T4 energy (|ΔE/E| < 0.2)"}


def main():
    pnd = np.load(f"{W}/round2/displays/pandora_event_preds.npz")
    out = {}
    for task in ("evt_nu", "flavor", "energy"):
        g, y, preds, correct = event_level(task, pnd)
        algs = list(correct)
        n_ok = sum(correct[a].astype(int) for a in algs)
        hard = n_ok <= 1
        d = dict(n_alg=len(algs), n_events=len(g), hist=np.bincount(n_ok, minlength=len(algs) + 1).tolist(),
                 acc={a: float(correct[a].mean()) for a in algs},
                 acc_on_hard={a: float(correct[a][hard].mean()) for a in algs},
                 n_boundary=int((n_ok == len(algs) // 2).sum()),
                 acc_on_boundary={a: float(correct[a][n_ok == len(algs) // 2].mean()) for a in algs})
        if task != "energy":
            fm = np.stack([preds[a] for a in algs if not a.startswith("pandora")]).astype(int)
            maj = np.array([np.bincount(c, minlength=3).argmax() for c in fm.T])
            d["hard_patterns"] = {f"{int(t)}->{int(p)}": n for (t, p), n in
                                  Counter(zip(y[hard].astype(int), maj[hard])).most_common(6)}
            d["hard_truth"] = np.bincount(y[hard].astype(int), minlength=3).tolist()
            d["all_truth"] = np.bincount(y.astype(int), minlength=3).tolist()
        else:
            e = np.exp(y)
            d["hard_E_quantiles"] = np.quantile(e[hard], [0.1, 0.5, 0.9]).tolist()
            d["all_E_quantiles"] = np.quantile(e, [0.1, 0.5, 0.9]).tolist()
            fmm = np.mean([preds[a] for a in algs if not a.startswith("pandora")], 0)
            d["hard_frac_overestimated_by_FMs"] = float((fmm[hard] > y[hard]).mean())
        out[task] = d
        print(task, d["hist"], flush=True)
    g, f1 = hit_level()
    nu_frac = f1.pop("_nu_frac")
    algs = list(f1)
    ok = {a: f1[a] >= 0.8 for a in algs}
    n_ok = sum(ok[a].astype(int) for a in algs)
    hard = n_ok == 0
    bnd = (n_ok == 3) | (n_ok == 4)
    out["hit"] = dict(n_alg=len(algs), n_events=len(g), hist=np.bincount(n_ok, minlength=len(algs) + 1).tolist(),
                      acc={a: float(ok[a].mean()) for a in algs}, median_f1={a: float(np.median(f1[a])) for a in algs},
                      n_boundary=int(bnd.sum()), acc_on_boundary={a: float(ok[a][bnd].mean()) for a in algs},
                      hard_nu_frac_quantiles=np.quantile(nu_frac[hard], [0.1, 0.5, 0.9]).tolist(),
                      all_nu_frac_quantiles=np.quantile(nu_frac, [0.1, 0.5, 0.9]).tolist())
    print("hit", out["hit"]["hist"])
    json.dump(out, open(sys.argv[1] + ".json", "w"), indent=1)
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.6))
    for ax, task in zip(axs, ("hit", "evt_nu", "flavor", "energy")):
        h = np.array(out[task]["hist"]) / out[task]["n_events"]
        ax.bar(np.arange(len(h)), h, color="tab:blue")
        ax.set_title(TITLE[task], fontsize=9)
        ax.set_xlabel(f"# algorithms correct (of {out[task]['n_alg']})", fontsize=8)
        ax.set_xticks(np.arange(len(h)))
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("fraction of test events", fontsize=8)
    fig.tight_layout()
    fig.savefig(sys.argv[1] + ".png", dpi=200)


if __name__ == "__main__":
    main()
