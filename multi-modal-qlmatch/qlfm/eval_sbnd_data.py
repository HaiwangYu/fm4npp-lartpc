"""SBND real data with Wire-Cell labels (issue #14): run a finished multi-task model on beam-on / off-beam caches and
keep per-event aggregates instead of per-hit predictions (the samples hold ~1.8B hits). No retraining.

python -m qlfm.eval_sbnd_data --run RUN_DIR --corpus DIRS [--pattern GLOB] --out OUT_ROOT/<tag>/<chunk>  (QLFM_DETECTOR=sbnd)
python -m qlfm.eval_sbnd_data report OUT_ROOT       (each OUT_ROOT/<tag>: its chunks are concatenated, then scored)

Per event (events.npz, indexed like the corpus): the evt_nu logits (score = log-odds of class 1, which does not
saturate like the softmax), and charge-weighted (q_*) and
hit-count (n_*) sums of: all hits, model nu hits (sigmoid > 0.5), Wire-Cell nu-candidate footprint hits (hit_pnd_nu,
charge-light matched after the TGM/STM/light-mismatch veto), the same restricted to candidates passing the PR cosmic
tagger (hit_pnd_nu_ct0), and the model/WC overlaps. Plus brightest-flash time/PE and run/subrun/event.

The report compares, per run:
  * event level: WC selects an event if it has any WC nu hit; each model's p_nu cut is set to pass the same off-beam
    fraction as WC, and the beam-on pass rates are compared (also on events whose brightest flash is in the beam window);
  * hit level: on beam-on events with a WC candidate, the share of WC nu charge the model also tags (wc_in_model), the
    share of model nu charge inside the WC footprint (model_in_wc), and the same on the simulation test split
    (RUN/hit/preds_test.npz + the simulation cache) for the sim -> data change; and the model's nu share of all
    charge in beam-on vs off-beam.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from qlfm import geometry as G
from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.eval_perturb import load_run

SUMS = ("tot", "model", "wc", "wc0", "model_wc", "model_wc0")
EV_KEEP = ("run", "subrun", "event", "fl_t", "fl_pe", "wc_n_bundle", "wc_n_bundle_cosmic")


class DS(EventDataset):
    def __getitem__(self, i):
        out = super().__getitem__(i)
        s, j = self.c.locate(out["gidx"])
        sh = self.c.shards[s]
        h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
        out["hit_ct0"] = np.asarray(sh.hit("pnd_nu_ct0")[h0:h0 + hn])
        return out


def collate_ct0(items):
    b = collate(items)
    b["hit_ct0"] = torch.from_numpy(np.concatenate([it["hit_ct0"] for it in items]).astype(np.uint8))
    return b


@torch.no_grad()
def predict_corpus(run, corpus_dirs, out, bs=16, workers=6, pattern="*", max_events=None):
    t0 = time.time()
    dev = torch.device("cuda")
    a, ns, model, multitask = load_run(run, dev)
    assert multitask and G.DETECTOR == "sbnd"
    model.tasks, model.light_off_pass = ["hit", "evt_nu"], False
    corpus = Corpus(corpus_dirs, pattern)
    n = corpus.n if max_events is None else min(corpus.n, max_events)
    loader = torch.utils.data.DataLoader(DS(corpus, np.arange(n), truth=True), batch_size=bs, shuffle=False,
                                         collate_fn=collate_ct0, num_workers=workers, pin_memory=True)
    agg = {f"{w}_{k}": np.zeros(n) for w in ("q", "n") for k in SUMS}
    logit = np.full((n, 3), np.nan, np.float32)
    torch.manual_seed(0)
    model.eval()
    for b in loader:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b)
        g = b["gidx"].cpu().numpy()
        he, ne = b["hit_evt"], b["n_events"]
        s = torch.sigmoid(o["hit"].float()).reshape(-1) > 0.5
        w, w0 = b["hit_pnd_nu"] > 0, b["hit_ct0"] > 0
        masks = {"tot": torch.ones_like(s), "model": s, "wc": w, "wc0": w0, "model_wc": s & w, "model_wc0": s & w0}
        q = b["hit_q"].clamp(min=0)
        for k, m in masks.items():
            for pre, val in (("q", q * m), ("n", m.float())):
                acc = torch.zeros(ne, device=dev, dtype=torch.float64).index_add_(0, he, val.double())
                agg[f"{pre}_{k}"][g] = acc.cpu().numpy()
        logit[g] = o["evt_nu"].float().cpu().numpy()
    assert not np.isnan(logit).any(), "events without a prediction"
    ev = {k: corpus.ev_array(k)[:n] for k in EV_KEEP}
    ev["sample"] = np.repeat([os.path.basename(os.path.dirname(sh.dir)) for sh in corpus.shards], corpus.sizes)[:n]
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "events.npz"), logit=logit, **agg, **{k: np.asarray(v) for k, v in ev.items()})
    meta = dict(run=run, corpus=corpus_dirs, pattern=pattern, n_events=int(n), backbone=ns.backbone, minutes=(time.time() - t0) / 60)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


# ---------------- report ----------------
def _share(num, den):
    return float(num.sum() / max(den.sum(), 1e-9))


def sim_reference(run):
    """Model vs WC hit agreement on the simulation test split (T1-eligible events with a WC candidate)."""
    a = json.load(open(os.path.join(run, "metrics.json")))
    c = Corpus(a["corpus"])
    z = np.load(os.path.join(run, "hit", "preds_test.npz"), mmap_mode="r")
    g, he = np.asarray(z["gidx"]), np.asarray(z["hit_evt"])
    pred, qz = np.asarray(z["pred"]), np.asarray(z["hit_q"], np.float32)
    starts, ends = np.searchsorted(he, g), np.searchsorted(he, g, side="right")
    s_of, l_of = c.locate(g)
    S = {k: 0.0 for k in ("q_wc", "q_model_wc", "q_model", "q_model_in_any", "q_wc0", "q_model_wc0")}
    n_ev = 0
    for i, (s, l) in enumerate(zip(s_of, l_of)):
        sh = c.shards[s]
        o, n = int(sh.ev("hit_off")[l]), int(sh.ev("hit_cnt")[l])
        a0, b0 = starts[i], ends[i]
        assert b0 - a0 == n
        q = np.clip(qz[a0:b0], 0, None).astype(np.float64)
        m = pred[a0:b0] > 0.5
        w, w0 = np.asarray(sh.hit("pnd_nu")[o:o + n]) > 0, np.asarray(sh.hit("pnd_nu_ct0")[o:o + n]) > 0
        if not w.any():
            continue
        n_ev += 1
        S["q_wc"] += (q * w).sum(); S["q_model_wc"] += (q * (m & w)).sum(); S["q_model"] += (q * m).sum()
        S["q_wc0"] += (q * w0).sum(); S["q_model_wc0"] += (q * (m & w0)).sum()
    return dict(n_events=n_ev, wc_in_model=S["q_model_wc"] / S["q_wc"], model_in_wc=S["q_model_wc"] / S["q_model"],
                wc0_in_model=S["q_model_wc0"] / max(S["q_wc0"], 1e-9))


def report_one(out, with_sim=True):
    chunks = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(out, "*", "events.npz")))
    metas = [json.load(open(os.path.join(c, "meta.json"))) for c in chunks]
    assert len({m["run"] for m in metas}) == 1, out
    meta = metas[0]
    parts = [dict(np.load(os.path.join(c, "events.npz"), allow_pickle=True)) for c in chunks]
    E = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    smp = E["sample"]
    on, off = smp == "bnb-opendata", smp == "offbeam-intime-run1"
    inwin = (E["fl_t"] >= G.BEAM_WINDOW_US[0]) & (E["fl_t"] < G.BEAM_WINDOW_US[1])
    lg = E["logit"].astype(np.float64)
    p = lg[:, 1] - np.logaddexp(lg[:, 0], lg[:, 2])           # nu log-odds (p > 0  <=>  softmax p_nu > 0.5)
    res = dict(run=meta["run"], chunks=len(chunks), n_on=int(on.sum()), n_off=int(off.sum()))
    for lab in ("wc", "wc0"):
        sel = E[f"n_{lab}"] > 0
        for cut_name, cut in (("all", np.ones_like(on)), ("beamwin", inwin)):
            o, f = on & cut, off & cut
            wc_on, wc_off = float(sel[o].mean()), float(sel[f].mean())
            thr = float(np.quantile(p[f], 1 - wc_off))
            res[f"evt/{lab}/{cut_name}"] = dict(n_on=int(o.sum()), n_off=int(f.sum()), wc_on=wc_on, wc_off=wc_off,
                                                model_on=float((p[o] > thr).mean()), model_off=float((p[f] > thr).mean()),
                                                thr=thr, model_on_p50=float((p[o] > 0).mean()),
                                                model_off_p50=float((p[f] > 0).mean()))
        m = on & sel                                       # hit level on beam-on events with a WC candidate
        res[f"hit/{lab}/beam_on"] = dict(n_events=int(m.sum()),
                                         wc_in_model=_share(E[f"q_model_{lab}"][m], E[f"q_{lab}"][m]),
                                         model_in_wc=_share(E[f"q_model_{lab}"][m], E["q_model"][m]))
    for nm, msk in (("beam_on", on), ("off_beam", off)):
        res[f"nu_charge_share/{nm}"] = dict(model=_share(E["q_model"][msk], E["q_tot"][msk]),
                                            wc=_share(E["q_wc"][msk], E["q_tot"][msk]),
                                            wc0=_share(E["q_wc0"][msk], E["q_tot"][msk]))
    if with_sim:
        res["hit/sim_test"] = sim_reference(meta["run"])
    json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1)
    return res


def main(argv=None):
    if argv and argv[0] == "report":
        rows = []
        for out in sorted(d for d in glob.glob(os.path.join(argv[1], "*")) if os.path.isdir(d)):
            r = report_one(out)
            rows.append(r)
            print(json.dumps(r), flush=True)
        with open(os.path.join(argv[1], "report_all.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*", help="shard glob inside each corpus dir (split a sample into chunks)")
    ap.add_argument("--max-events", type=int, default=None, help="smoke tests")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=None, help="accepted for run_queue.sh; unused")
    a = ap.parse_args(argv)
    predict_corpus(a.run, a.corpus, a.out, a.bs, a.workers, a.pattern, a.max_events)
    json.dump({"done": True}, open(os.path.join(a.out, "metrics.json"), "w"))     # run_queue.sh completion marker


if __name__ == "__main__":
    main(sys.argv[1:])
