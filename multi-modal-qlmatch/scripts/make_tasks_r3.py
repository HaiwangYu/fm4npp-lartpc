#!/usr/bin/env python
"""Round-3 (fairness controls, issue #5) downstream task lines for scripts/run_queue.sh.

Sets (combine freely):
  xf    strong late fusion (--late-light 2): frozen at 1 % and 10 % labels, all tasks   -> <bb>/<task>/frozen_xf/
  hit   T1 per-hit input: common raw-hit adapter for every backbone (frozen_raw); qlfm without its hit
        embedding (frozen_nohit); frozen at 1 %, R2P late light
  r2p   the full R2P matrix (same as scripts/make_tasks_r2.py) for new backbones: FM4NPP without the hit cap,
        qlfm-M C, FM4NPP m4

Usage: make_tasks_r3.py OUT_ROOT SET BACKBONE... > tasks.txt
"""
import os
import sys

W = os.path.expandvars("$QLFM_WORK")
P = f"{W}/pre"
TASKS = ("evt_nu", "flavor", "energy", "hit")
NOCAP = " --max-hits 100000 --accum 4"     # uncapped Mamba2: >~1e9-element batches crash the kernel (int32)
# key: (interpreter prefix, --backbone, checkpoint, flags for every run, extra flags for fine-tuning)
BACKBONES = {
    "qlfm_M_CL": ("", "qlfm", f"{P}/m_cl_f0.90/fm.pt", "", ""),
    "qlfm_S_CL": ("", "qlfm", f"{P}/cl_f0.90/fm.pt", "", ""),
    "qlfm_S_C": ("", "qlfm", f"{P}/c_f0.90/fm.pt", "", ""),
    "qlfm_M_C": ("", "qlfm", f"{P}/m_c_f0.90/fm.pt", "", ""),
    "panda": ("@pimm ", "panda", f"{P}/panda", " --accum 4", " --accum 8"),
    "polarmae": ("@pimm ", "polarmae", f"{P}/polarmae", "", " --accum 4"),
    "fm4npp": ("@mamba ", "fm4npp", f"{P}/fm4npp/fm.pt", NOCAP, " --accum 8"),
    "fm4npp_m4": ("@mamba ", "fm4npp", f"{P}/fm4npp_m4/fm.pt", NOCAP, " --accum 8"),
}


def runs(which, key):
    """yield (mode, out_tag, label_frac, seed, late_light, extra)"""
    if which == "xf":
        for lf in (0.01, 0.1):
            for t in TASKS:
                yield t, "frozen", "frozen_xf", lf, 0, 2, ""
    elif which == "hit":
        yield "hit", "frozen", "frozen_raw", 0.01, 0, 1, " --hit-feat raw"
        if key.startswith("qlfm"):
            yield "hit", "frozen", "frozen_nohit", 0.01, 0, 1, " --hit-feat none"
    elif which == "r2p":
        for t in TASKS:
            for seed in (0, 1):
                yield t, "frozen", "frozen", 0.001, seed, 1, ""
            for lf in (0.01, 0.1):
                yield t, "frozen", "frozen", lf, 0, 1, ""
            for lf in (0.001, 0.01):
                yield t, "finetune", "finetune", lf, 0, 1, " --layer-decay 0.9"
            yield t, "frozen", "frozen_nolight", 0.01, 0, 0, ""
    else:
        raise ValueError(which)


def main():
    root, which, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
    lines = []
    for key in keys:
        prefix, bb, ck, flags, ft_flags = BACKBONES[key]
        for t, mode, tag, lf, seed, late, extra in runs(which, key):
            ev = " --n-test 3000 --n-val 1000" if t == "hit" else " --n-test 10000 --n-val 2000"
            f = flags + (ft_flags if mode == "finetune" else "")
            if mode == "finetune" and "--accum" in ft_flags:
                f = f.replace(" --accum 4", "") if " --accum 8" in ft_flags else f
            line = (f"{prefix}qlfm.train_downstream --task {t} --mode {mode} --backbone {bb} --fm {ck} "
                    f"--label-frac {lf:g} --seed {seed} --late-light {late}{extra}{ev} --max-steps 3000{f} --resume 1 "
                    f"--out {root}/{key}/{t}/{tag}/f{lf:g}_s{seed}")
            long_run = lf >= 0.01
            lines.append((0 if long_run else 1, line))
    for _, l in sorted(lines, key=lambda x: x[0]):
        print(l)


if __name__ == "__main__":
    main()
