#!/usr/bin/env python
"""Round-4 (issue #6) T5 charge -> light task lines for scripts/run_queue.sh.

  frozen  : every pretrained checkpoint of the pretraining-size scan x label fractions 0.1/1/10/100 %
  finetune: the three round-4 encoders (90 % pretraining) at 1 % and 10 %
  scratch : qlfm-M architecture without pretraining at 1/10/100 % (reference)
  zeroshot: the charge+light MAE decoder itself (qlfm CL checkpoints), no labelled training

Usage: make_tasks_r4.py OUT_ROOT [--only KEY...] > tasks.txt   (OUT_ROOT/<key>/light/<mode>/f<lf>_s0)
"""
import os
import sys

W = os.path.expandvars("$QLFM_WORK")
P = f"{W}/pre"
LF = (0.001, 0.01, 0.1, 1.0)
COMMON = " --n-test 10000 --n-val 2000 --resume 1"
# step cap per label fraction: at >= 10 % labels a 6000-step cap (192k events) would make the 10 % and 100 % runs
# see the same number of events, so the label-size scan would measure compute rather than data
STEPS = {0.001: 6000, 0.01: 6000, 0.1: 24000, 1.0: 24000}
FT_STEPS = {0.01: 6000, 0.1: 12000}
M_ARCH = " --d 384 --depth 12 --heads 6 --light 0 --max-tokens 4096"
# key: (interpreter prefix, --backbone, checkpoint, flags, fine-tune flags, MAE ckpt for zero-shot or None)
CKPTS = {}
for f in ("0.01", "0.10", "0.50", "0.90"):
    CKPTS[f"qlfm_S_CL_pre{f}"] = ("", "qlfm", f"{P}/cl_f{f}/fm.pt", "", "", f"{P}/cl_f{f}/ckpt.pt")
    CKPTS[f"qlfm_S_C_pre{f}"] = ("", "qlfm", f"{P}/c_f{f}/fm.pt", "", "", None)
for f in ("0.10", "0.90"):
    CKPTS[f"qlfm_M_CL_pre{f}"] = ("", "qlfm", f"{P}/m_cl_f{f}/fm.pt", "", "", f"{P}/m_cl_f{f}/ckpt.pt")
    CKPTS[f"qlfm_M_C_pre{f}"] = ("", "qlfm", f"{P}/m_c_f{f}/fm.pt", "", "", None)
CKPTS["panda_pre0.10"] = ("@pimm ", "panda", f"{P}/panda_f0.10", " --accum 4", " --accum 8", None)
CKPTS["panda_pre0.90"] = ("@pimm ", "panda", f"{P}/panda", " --accum 4", " --accum 8", None)
FT = ("qlfm_M_CL_pre0.90", "qlfm_M_C_pre0.90", "panda_pre0.90")


def main():
    root = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    lines = []
    for key, (prefix, bb, ck, flags, ft_flags, mae) in CKPTS.items():
        if only and key not in only:
            continue
        base = f"{prefix}qlfm.train_downstream --task light --backbone {bb} --fm {ck} --seed 0"
        if mae:
            lines.append((0, f"qlfm.t5_zeroshot --ckpt {mae} --n-test 10000 --out {root}/{key}/light/zeroshot/f0_s0"))
        for lf in LF:
            lines.append((1 if lf < 0.01 else 2, f"{base} --mode frozen --label-frac {lf:g} --max-steps {STEPS[lf]}{COMMON}{flags} "
                                                  f"--out {root}/{key}/light/frozen/f{lf:g}_s0"))
        if key in FT:
            for lf in (0.01, 0.1):
                f = ft_flags if ft_flags else flags
                lines.append((3, f"{base} --mode finetune --label-frac {lf:g} --layer-decay 0.9 --max-steps {FT_STEPS[lf]}{COMMON}{f} "
                                 f"--out {root}/{key}/light/finetune/f{lf:g}_s0"))
    if not only:
        for lf in (0.01, 0.1, 1.0):
            lines.append((3, f"qlfm.train_downstream --task light --mode scratch --label-frac {lf:g} --max-steps {STEPS[lf]} --seed 0{M_ARCH}"
                             f"{COMMON} --out {root}/scratch_M/light/scratch/f{lf:g}_s0"))
    for _, l in sorted(lines, key=lambda x: x[0]):
        print(l)


if __name__ == "__main__":
    main()
