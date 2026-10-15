#!/usr/bin/env python
"""Generate downstream task files (one `qlfm.train_downstream ...` command per line) for run_queue.sh.

Examples:
  make_tasks.py e3 --fm-c RUN_C/fm.pt --fm-cl RUN_CL/fm.pt --out-root $QLFM_WORK/e3 > tasks_e3.txt
  make_tasks.py e4 --fm-cl-frac 0.01:RUN/fm.pt,0.10:RUN2/fm.pt --out-root $QLFM_WORK/e4 > tasks_e4.txt
"""
import argparse

TASKS = ("evt_nu", "flavor", "energy", "hit")
FRACS = (0.001, 0.01, 0.1, 1.0)


def seeds_for(frac):
    return (0, 1, 2) if frac <= 0.01 else (0,)


def line(task, mode, frac, seed, out, fm=None, light=None, extra=""):
    s = f"qlfm.train_downstream --task {task} --mode {mode} --label-frac {frac:g} --seed {seed}"
    if fm:
        s += f" --fm {fm}"
    if light is not None:
        s += f" --light {light}"
    if task == "hit":
        s += " --n-test 3000 --n-val 1000"
    return s + (" " + extra if extra else "") + f" --out {out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e3", "e4"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fm-c")
    ap.add_argument("--fm-cl")
    ap.add_argument("--fm-cl-frac", default="", help="comma list frac:path for E4")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--fracs", default=",".join(f"{f:g}" for f in FRACS))
    ap.add_argument("--modes", default="scratch,frozen,finetune")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    fracs = [float(f) for f in a.fracs.split(",")]
    out = []
    if a.exp == "e1":  # scratch sanity, C and CL, one label budget
        for t in tasks:
            for inp, light in (("C", 0), ("CL", 1)):
                for f in fracs:
                    out.append(line(t, "scratch", f, 0, f"{a.out_root}/{t}/scratch_{inp}/f{f:g}_s0", light=light, extra=a.extra))
    elif a.exp == "e3":
        # order: most informative / cheapest first (small label budgets, frozen)
        for f in fracs:
            for s in seeds_for(f):
                for t in tasks:
                    for mode in a.modes.split(","):
                        for inp, fm, light in (("C", a.fm_c, 0), ("CL", a.fm_cl, 1)):
                            o = f"{a.out_root}/{t}/{mode}_{inp}/f{f:g}_s{s}"
                            out.append(line(t, mode, f, s, o, fm=None if mode == "scratch" else fm,
                                            light=light if mode == "scratch" else None, extra=a.extra))
    elif a.exp == "e4":
        for item in a.fm_cl_frac.split(","):
            pf, path = item.split(":", 1)
            for t in tasks:
                for f in fracs:
                    for mode in a.modes.split(","):
                        out.append(line(t, mode, f, 0, f"{a.out_root}/pre{pf}/{t}/{mode}/f{f:g}_s0", fm=path, extra=a.extra))
    print("\n".join(out))


if __name__ == "__main__":
    main()
