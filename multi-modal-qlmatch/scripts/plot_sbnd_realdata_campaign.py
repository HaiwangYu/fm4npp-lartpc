#!/usr/bin/env python
"""Figures for the SBND real-data campaign slides (#14), from the first-look and model-vs-Wire-Cell outputs.

Usage: python scripts/plot_sbnd_realdata_campaign.py OUT_DIR
Reads $QLFM_WORK/sbnd_data/firstlook.json and $QLFM_WORK/sbnd_data/eval/report_all.jsonl.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

W = os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # categorical slots 1-5, fixed order
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3dd"
TICK_TO_CM = 0.07815
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 220})

fl = json.load(open(os.path.join(W, "sbnd_data", "firstlook.json")))

# 1. hit-width ratio data / sim vs drift, per plane
e = np.array(fl["drift_edges_ticks"])
x = 0.5 * (e[1:] + e[:-1]) * TICK_TO_CM
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for p, (name, c) in enumerate(zip(("U", "V", "Y"), C)):
    ax.plot(x, fl["on"][f"rms_ratio_to_sim_p{p}_drift"], "-o", color=c, lw=2, ms=5, label=f"{name}, beam-on")
    ax.plot(x, fl["off"][f"rms_ratio_to_sim_p{p}_drift"], "--s", color=c, lw=1.5, ms=4, mfc="white",
            label=f"{name}, off-beam")
ax.axhline(1.0, color=MUTED, lw=1)
ax.set_xlabel("drift distance from the anode (cm)")
ax.set_ylabel("median hit RMS, data / sim")
ax.set_ylim(0.94, 1.02)
ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "width_ratio.png"))
plt.close(fig)

# 2. light time profile on the beam clock (shape: fraction of PE per bin)
t = np.array(fl["t_edges_us"])
tc = 0.5 * (t[1:] + t[:-1])
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.axvspan(*fl["beam_window_us"], color="#f1f0ea", zorder=0, label="beam window 0-3 µs")
for (k, lab), c, ls in zip((("on", "beam-on"), ("off", "off-beam"), ("sim", "BNB CV sim")), C, ("-", "--", ":")):
    h = np.array(fl[k]["light_profile_pe_per_event"])
    ax.plot(tc, h / h.sum(), ls, color=c, lw=2, label=lab, drawstyle="steps-mid")
ax.set_xlim(-4, 8)
ax.set_xlabel("optical-hit time, beam clock (µs)")
ax.set_ylabel("fraction of PE per 0.25 µs")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "light_profile.png"))
plt.close(fig)

# 3 + 4: model results
rows = [json.loads(l) for l in open(os.path.join(W, "sbnd_data", "eval", "report_all.jsonl"))]
FAM = [("sbndpre_M_CL/finetune", "SBND-pre CL, fine-tuned"), ("sbndpre_M_C_xf/finetune", "SBND-pre C+XF"),
       ("ubpre_M_CL/finetune", "MicroBooNE-pre CL"), ("sbndpre_M_CL/frozen", "SBND-pre CL, frozen"),
       ("scratch_M_CL/finetune", "no pretraining")]
MK = ["o", "s", "D", "^", "v"]
fam = lambda r: next(i for i, (k, _) in enumerate(FAM) if k in r["run"])

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, lw=1, ls="--")
ax.text(0.95, 0.92, "data = sim", color=MUTED, fontsize=7, ha="right", rotation=38)
for i, (k, lab) in enumerate(FAM):
    rr = [r for r in rows if fam(r) == i]
    ax.plot([r["hit/sim_test"]["wc_in_model"] for r in rr], [r["hit/wc/beam_on"]["wc_in_model"] for r in rr],
            MK[i], color=C[i], ms=7, mec="white", mew=1, label=lab, linestyle="none")
ax.set_xlim(0.72, 0.95)
ax.set_ylim(0.3, 0.95)
ax.set_xlabel("simulation test: share of WC ν charge also tagged")
ax.set_ylabel("beam-on data: same share")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "agreement.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 2.9))
wc = rows[0]["evt/wc/all"]["wc_on"]
labels, vals = ["Wire-Cell"], [[wc]]
for i, (k, lab) in enumerate(FAM):
    labels.append(lab)
    vals.append([r["evt/wc/all"]["model_on"] for r in rows if fam(r) == i])
y = np.arange(len(labels))[::-1]
for j, (yy, v) in enumerate(zip(y, vals)):
    col = INK if j == 0 else C[j - 1]
    m = float(np.mean(v))
    ax.barh(yy, m, height=0.55, color=col, alpha=0.9 if j else 0.75)
    if len(v) > 1:
        ax.plot([min(v), max(v)], [yy, yy], color=INK, lw=1.2)
    ax.text(max(v) + 0.01, yy, f"{100 * m:.1f}%" + (f" ({100 * min(v):.1f}-{100 * max(v):.1f})" if len(v) > 1 else ""),
            va="center", fontsize=7, color=INK)
off = rows[0]["evt/wc/all"]["wc_off"]
ax.axvline(off, color=MUTED, lw=1, ls=":")
ax.text(off + 0.005, y[0] + 0.45, f"off-beam {100 * off:.1f}%", fontsize=7, color=MUTED)
ax.set_yticks(y, labels, fontsize=7.5)
ax.set_xlim(0, 0.62)
ax.set_ylim(y[-1] - 0.6, y[0] + 0.8)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("beam-on pass rate, off-beam fixed at 5.8%")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "event_level.png"))
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
