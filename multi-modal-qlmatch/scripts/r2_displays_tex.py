#!/usr/bin/env python
"""Write the LaTeX appendix (one display per page) for the round-2 tech note.
Usage: r2_displays_tex.py SELECTION.json IMG_DIR(relative to the .tex) OUT.tex"""
import json
import sys

TITLE = {"hit": "T1: hit-level $\\nu$/cosmic", "evt_nu": "T2: event-level $\\nu$ presence",
         "flavor": "T3: flavor", "energy": "T4: neutrino energy"}
sel, img, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
lines = []
for i, task in enumerate(("hit", "evt_nu", "flavor", "energy")):
    lines.append(("" if i == 0 else "\\clearpage") + f"\\subsection{{{TITLE[task]}}}")
    cnt = {"easy": 0, "boundary": 0, "hard": 0}
    for it in sel[task]:
        cnt[it["kind"]] += 1
        name = f"{task}_{it['kind']}{cnt[it['kind']]:02d}"
        lines.append(f"\\begin{{center}}\\includegraphics[height=0.86\\textheight,width=\\textwidth,keepaspectratio]"
                     f"{{{img}/{name}.png}}\\end{{center}}\\clearpage")
open(out, "w").write("\n".join(lines) + "\n")
