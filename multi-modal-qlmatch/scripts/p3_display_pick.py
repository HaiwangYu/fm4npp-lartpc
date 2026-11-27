#!/usr/bin/env python
"""Pick SBND data events where a model and Wire-Cell disagree, for scripts/sbnd_event_displays.py --events (#16).

Reads the per-event aggregates of qlfm.eval_sbnd_data (EVAL_DIR/{on,off}_m*/events.npz), keeps the held-out test
files, maps events to the full data corpus by (sample, run, subrun, event) and writes groups of N events:
  on_extra   beam-on: model nu charge outside the WC footprint (q_model - q_model_wc)
  on_missed  beam-on with a WC candidate: WC nu charge the model does not tag (q_wc - q_model_wc)
  on_evt     beam-on selected by the model at WC's off-beam pass rate, no WC candidate (event score)
  off_nu     off-beam: charge the model calls nu (q_model)
Each group draws N at random from the top TOP events of its measure.

python scripts/p3_display_pick.py EVAL_DIR OUT.json [--n 10] [--top 200] [--seed 0]
"""
import argparse
import glob
import json
import os

import numpy as np

from qlfm.data.dataset import Corpus
from qlfm.eval_sbnd_data import heldout

W = os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
DATA = f"{W}/cache/sbnd_data_m/bnb-opendata,{W}/cache/sbnd_data_m/offbeam-intime-run1"


def load(ev_dir, side):
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(ev_dir, f"{side}_m*", "events.npz")))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    on, off = load(a.eval_dir, "on"), load(a.eval_dir, "off")
    lo = lambda d: d["logit"][:, 1] - np.logaddexp(d["logit"][:, 0], d["logit"][:, 2])
    off_all = off
    on = {k: v[heldout(on["run"], on["subrun"], on["event"])] for k, v in on.items()}
    off = {k: v[heldout(off["run"], off["subrun"], off["event"])] for k, v in off.items()}
    # event threshold: the model passes off-beam (all off-beam events) at WC's off-beam candidate rate
    wc_rate = float((off_all["n_wc"] > 0).mean())
    thr = float(np.quantile(lo(off_all), 1 - wc_rate))
    groups = {
        "on_extra": ("on", on, on["q_model"] - on["q_model_wc"], None,
                     "model ν charge outside the Wire-Cell footprint"),
        "on_missed": ("on", on, on["q_wc"] - on["q_model_wc"], on["n_wc"] > 0,
                      "Wire-Cell ν charge the model does not tag"),
        "on_evt": ("on", on, lo(on), (lo(on) > thr) & (on["n_wc"] == 0),
                   "selected by the model, no Wire-Cell candidate"),
        "off_nu": ("off", off, off["q_model"], None, "off-beam charge the model calls ν"),
    }
    dc = Corpus(DATA)
    key = lambda s, r, sr, e: (np.asarray(s, np.int64) << 60) | (np.asarray(r, np.int64) << 40) | \
        (np.asarray(sr, np.int64) << 20) | np.asarray(e, np.int64)
    ck = key(dc.sample_id(), dc.ev_array("run"), dc.ev_array("subrun"), dc.ev_array("event"))
    order = np.argsort(ck)
    out = {}
    for g, (kind, d, score, mask, title) in groups.items():
        idx = np.flatnonzero(mask if mask is not None else np.ones(len(score), bool))
        idx = idx[np.argsort(-score[idx])][:a.top]
        pick = np.sort(rng.choice(idx, min(a.n, len(idx)), replace=False))
        k = key(0 if kind == "on" else 1, d["run"][pick], d["subrun"][pick], d["event"][pick])
        j = np.searchsorted(ck[order], k)
        assert (ck[order][j] == k).all(), g
        out[g] = dict(kind=kind, title=f"{title} (P3 + data-like widths)", gidx=order[j].tolist(),
                      n_candidates=int(len(np.flatnonzero(mask if mask is not None else np.ones(len(score), bool)))),
                      score=score[pick].round(1).tolist())
        print(g, kind, len(pick), "of top", len(idx), "| candidates", out[g]["n_candidates"])
    out["_meta"] = dict(eval_dir=a.eval_dir, wc_offbeam_rate=wc_rate, thr_logodds=thr, n_on=int(len(on["run"])),
                        n_off=int(len(off["run"])), top=a.top, seed=a.seed)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
