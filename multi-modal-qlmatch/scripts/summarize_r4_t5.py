#!/usr/bin/env python
"""Round 4 (issue #6) T5 charge -> light: LaTeX table fragments + scaling figure.
Planned runs without metrics.json are printed as \\pend, so the tech note can be rebuilt as results arrive.

Reference points: zero-shot (the charge+light MAE decoder, no labels), scratch (qlfm-M architecture, no
pretraining) and a constant predictor (per-PMT, per-feature mean of the measured test light; optimistic, it uses
the test set itself).

Usage: summarize_r4_t5.py T5_ROOT TEX_DIR FIG_PNG
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qlfm.evaluate import light_metrics  # noqa: E402

LF = ("0.001", "0.01", "0.1", "1")
LFNAME = {"0.001": r"0.1\%", "0.01": r"1\%", "0.1": r"10\%", "1": r"100\%"}
# (label, family, pretrain fraction, run key)
ENC = [("qlfm-S CL", "S CL", 0.01, "qlfm_S_CL_pre0.01"), ("qlfm-S CL", "S CL", 0.10, "qlfm_S_CL_pre0.10"),
       ("qlfm-S CL", "S CL", 0.50, "qlfm_S_CL_pre0.50"), ("qlfm-S CL", "S CL", 0.90, "qlfm_S_CL_pre0.90"),
       ("qlfm-S C", "S C", 0.01, "qlfm_S_C_pre0.01"), ("qlfm-S C", "S C", 0.10, "qlfm_S_C_pre0.10"),
       ("qlfm-S C", "S C", 0.50, "qlfm_S_C_pre0.50"), ("qlfm-S C", "S C", 0.90, "qlfm_S_C_pre0.90"),
       ("qlfm-M CL", "M CL", 0.10, "qlfm_M_CL_pre0.10"), ("qlfm-M CL", "M CL", 0.90, "qlfm_M_CL_pre0.90"),
       ("qlfm-M C", "M C", 0.10, "qlfm_M_C_pre0.10"), ("qlfm-M C", "M C", 0.90, "qlfm_M_C_pre0.90"),
       ("Panda", "Panda", 0.10, "panda_pre0.10"), ("Panda", "Panda", 0.90, "panda_pre0.90")]
STYLE = {"M CL": ("tab:red", "o"), "M C": ("tab:blue", "s"), "Panda": ("tab:green", "^"),
         "S CL": ("tab:orange", "o"), "S C": ("tab:cyan", "s")}
FT = ("qlfm_M_CL_pre0.90", "qlfm_M_C_pre0.90", "panda_pre0.90")
COLS = [("mse", "MSE", False, "{:.3f}"), ("tot_resolution", r"$\sigma_{\rm tot}$", False, "{:.2f}"),
        ("tot_bias", "bias", None, "{:+.2f}"), ("pattern_tv_med", "TV$_{\\rm PMT}$", False, "{:.2f}"),
        ("corr_log_total", r"$\rho$", True, "{:.2f}")]


def get(root, key, mode, lf):
    p = f"{root}/{key}/light/{mode}/f{lf}_s0/metrics.json"
    return json.load(open(p))["test"] if os.path.exists(p) else None


def fmt(m, k, f):
    return r"\pend" if m is None else (f.format(m[k]) if k in m and np.isfinite(m[k]) else "--")


def constant_ref(root):
    for p in sorted(glob.glob(f"{root}/*/light/zeroshot/f0_s0/preds_test.npz")):
        t = np.load(p)["light_tgt"].astype(np.float32)
        return light_metrics({"pred": np.broadcast_to(t.mean(0, keepdims=True), t.shape), "light_tgt": t})
    return None


def main():
    root, tex, fig = sys.argv[1:4]
    os.makedirs(tex, exist_ok=True)
    const = constant_ref(root)
    json.dump(const, open(f"{root}/t5_constant.json", "w"), indent=1)
    n_plan = n_done = 0

    # table 1: frozen MSE vs label fraction (rows: encoder x pretrain fraction) + zero-shot column
    lines = []
    for lab, fam, pf, key in ENC:
        row = [lab, f"{pf:.0%}".replace("%", r"\%")]
        for lf in LF:
            m = get(root, key, "frozen", lf)
            n_plan += 1
            n_done += m is not None
            row.append(fmt(m, "mse", "{:.3f}"))
        zs = get(root, key, "zeroshot", "0")
        if "CL" in fam:
            n_plan += 1
            n_done += zs is not None
        row.append(fmt(zs, "mse", "{:.3f}") if "CL" in fam else "--")
        lines.append(" & ".join(row) + r" \\")
    sc = [get(root, "scratch_M", "scratch", lf) for lf in LF]
    n_plan += 3
    n_done += sum(m is not None for m in sc[1:])
    lines.append(r"\midrule")
    lines.append(" & ".join(["scratch (qlfm-M arch.)", "--", "--"] + [fmt(m, "mse", "{:.3f}") for m in sc[1:]] + ["--"])
                 + r" \\")
    if const:
        lines.append(" & ".join([r"constant (test mean)", "--"] + [f"{const['mse']:.3f}"] * 4 + ["--"]) + r" \\")
    open(f"{tex}/t5_frozen.tex", "w").write("\n".join(lines) + "\n")

    # table 2: physics summaries for the round-4 encoders (90 % pretraining): frozen / fine-tune / zero-shot / scratch
    lines = []
    rows = []
    for lab, key in (("qlfm-M CL", "qlfm_M_CL_pre0.90"), ("qlfm-M C", "qlfm_M_C_pre0.90"), ("Panda", "panda_pre0.90")):
        if key == "qlfm_M_CL_pre0.90":
            rows.append((lab, "zero-shot", "--", get(root, key, "zeroshot", "0")))
        for lf in ("0.01", "0.1", "1"):
            rows.append((lab, "frozen", LFNAME[lf], get(root, key, "frozen", lf)))
        for lf in ("0.01", "0.1"):
            n_plan += 1
            m = get(root, key, "finetune", lf)
            n_done += m is not None
            rows.append((lab, "fine-tune", LFNAME[lf], m))
    for lf in ("0.01", "0.1", "1"):
        rows.append(("scratch", "scratch", LFNAME[lf], get(root, "scratch_M", "scratch", lf)))
    if const:
        rows.append(("constant", "test mean", "--", const))
    prev = None
    for lab, mode, lf, m in rows:
        if prev is not None and lab != prev:
            lines.append(r"\midrule")
        prev = lab
        lines.append(" & ".join([lab, mode, lf] + [fmt(m, k, f) for k, _, _, f in COLS]) + r" \\")
    open(f"{tex}/t5_physics.tex", "w").write("\n".join(lines) + "\n")
    open(f"{tex}/t5_status.tex", "w").write(f"\\newcommand{{\\tdone}}{{{n_done}}}\\newcommand{{\\tplan}}{{{n_plan}}}\n")

    # figure: (a) MSE vs label fraction, (b) MSE vs pretraining fraction, (c) total-PE resolution vs label fraction
    f, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.array([float(v) for v in LF]) * 100
    for fam, (c, mk) in STYLE.items():
        for lab, fam_, pf, key in ENC:
            if fam_ != fam or pf != 0.90:
                continue
            for a, k in ((ax[0], "mse"), (ax[2], "tot_resolution")):
                y = [(get(root, key, "frozen", lf) or {}).get(k, np.nan) for lf in LF]
                a.plot(x, y, "-", color=c, marker=mk, label=f"{lab} frozen")
                if key in FT:
                    yf = [(get(root, key, "finetune", lf) or {}).get(k, np.nan) for lf in ("0.01", "0.1")]
                    a.plot(x[1:3], yf, "--", color=c, marker=mk, mfc="none", label=f"{lab} fine-tune")
                zs = get(root, key, "zeroshot", "0")
                if zs:
                    a.axhline(zs[k], color=c, ls=":", lw=1.2, label=f"{lab} zero-shot")
    for a, k in ((ax[0], "mse"), (ax[2], "tot_resolution")):
        a.plot(x[1:], [(m or {}).get(k, np.nan) for m in sc[1:]], "-", color="k", marker="x", label="scratch (M arch.)")
        if const and k in const:
            a.axhline(const[k], color="0.5", ls="-.", lw=1, label="constant (test mean)")
        a.set_xscale("log")
        a.set_xlim(0.07, 150)
        a.set_xlabel("labelled training events (% of train split)")
    ax[0].set_ylabel("test MSE (log1p PE features, 32 PMT x 17)")
    ax[2].set_ylabel(r"total-PE resolution $(q_{84}-q_{16})/2$ of $\Sigma_{pred}/\Sigma_{meas}-1$")
    ax[0].set_title("T5 vs labelled data (90 % pretraining)", fontsize=10)
    ax[2].set_title(r"T5 total-PE resolution (events with $\geq$200 PE)", fontsize=10)
    ax[0].legend(fontsize=6.5, ncol=2)
    for fam, (c, mk) in STYLE.items():
        for lf, ls in (("0.01", "-"), ("0.1", "--")):
            pts = [(pf, (get(root, key, "frozen", lf) or {}).get("mse", np.nan)) for _, fam_, pf, key in ENC if fam_ == fam]
            ax[1].plot([p * 100 for p, _ in pts], [v for _, v in pts], ls, color=c, marker=mk,
                       mfc=None if ls == "-" else "none", label=f"{fam}, {LFNAME[lf].replace(chr(92), '')} labels")
        if "CL" in fam:
            pts = [(pf, (get(root, key, "zeroshot", "0") or {}).get("mse", np.nan)) for _, fam_, pf, key in ENC if fam_ == fam]
            ax[1].plot([p * 100 for p, _ in pts], [v for _, v in pts], ":", color=c, marker="*", label=f"{fam}, zero-shot")
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.7, 130)
    ax[1].set_xlabel("pretraining data (% of pretrain split)")
    ax[1].set_ylabel("test MSE (frozen)")
    ax[1].set_title("T5 vs pretraining data", fontsize=10)
    ax[1].legend(fontsize=6.5, ncol=2)
    f.tight_layout()
    f.savefig(fig, dpi=150)
    print(f"done {n_done}/{n_plan}", "constant:", {k: round(v, 3) for k, v in (const or {}).items()})


if __name__ == "__main__":
    main()
