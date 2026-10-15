#!/usr/bin/env python
"""Data audit on a converted corpus: occupancy, timing, label statistics, classical baselines, event display.

Usage: audit.py CORPUS_DIR OUTDIR [--max-shards N]
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from qlfm import geometry as G
from qlfm.data.dataset import Corpus
from qlfm.data.labels import EV_KEYS, HIT_NU_EDEP_MEV, event_labels


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.mean(y) < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--max-shards", type=int, default=3)
    ap.add_argument("--splits", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = Corpus(a.corpus)
    ev = {k: c.ev_array(k) for k in EV_KEYS + ("hit_cnt", "oph_cnt", "pnd_score", "fl_t", "fl_pe", "nu_time", "nu_edep")}
    lab = event_labels(ev)
    rep = {"n_events": int(c.n), "n_shards": len(c.shards)}
    rep["hits_per_event_pct_1_50_99"] = np.percentile(ev["hit_cnt"], [1, 50, 99]).tolist()
    rep["ophits_per_event_pct_1_50_99"] = np.percentile(ev["oph_cnt"], [1, 50, 99]).tolist()
    in_av = G.in_volume(ev["nu_vtx"])
    rep["frac_vtx_in_AV"] = float(in_av.mean())
    rep["frac_vtx_in_FV"] = float(G.in_volume(ev["nu_vtx"], G.FV_MARGIN_CM).mean())
    ntot = ev["nu_nhits"].sum(1)
    rep["frac_no_nu_hits(<=5)"] = float((ntot <= 5).mean())
    for t, (m, y) in lab.items():
        d = {"n_eligible": int(m.sum())}
        if y is not None and t in ("evt_nu", "flavor"):
            d["class_counts"] = np.bincount(y[m].astype(int)).tolist()
        rep["task_" + t] = d
    # ambiguous T2 events (neither positive nor negative)
    rep["task_evt_nu"]["n_ambiguous"] = int((~lab["evt_nu"][0]).sum())

    # classical baselines for T2 (event-level), on eligible events
    m, y = lab["evt_nu"]
    s_pnd = ev["pnd_score"][m]
    rep["baseline_T2_auc_pandora_nu_score"] = auc(y[m], s_pnd)
    in_win = (ev["fl_t"] > 3.0) & (ev["fl_t"] < 6.0)
    s_light = np.where(in_win, np.log1p(ev["fl_pe"]), 0.0)[m]
    rep["baseline_T2_auc_beamflash_pe"] = auc(y[m], s_light)

    # timing: flash time - nu time for visible events
    v = lab["hit"][0] & (ev["fl_t"] > 0)
    dt = ev["fl_t"][v] - ev["nu_time"][v] / 1000
    rep["flash_minus_nu_time_us_pct_5_50_95"] = np.percentile(dt, [5, 50, 95]).tolist()

    # hit-level stats + Pandora T1 baseline on a few shards
    tq = tp = tpp = tpq = 0.0
    nu_frac = []
    toks = []
    for sh in c.shards[: a.max_shards]:
        q = np.asarray(sh.hit("integral"), dtype=np.float32)
        nu = np.asarray(sh.hit("nu_edep"), dtype=np.float32) > HIT_NU_EDEP_MEV
        pnd = np.asarray(sh.hit("pnd_nu")).astype(bool)
        cnt = np.asarray(sh.ev("hit_cnt"))
        evt = np.repeat(np.arange(sh.n), cnt)
        vis = event_labels({k: np.asarray(sh.ev(k)) for k in EV_KEYS})["hit"][0][evt]
        q, nu, pnd = q[vis], nu[vis], pnd[vis]
        tq += (q * nu).sum(); tp += (q * (nu & pnd)).sum(); tpp += (q * pnd).sum()
        nu_frac.append(nu.mean())
        pl = np.asarray(sh.hit("plane")).astype(np.int64)
        w = np.asarray(sh.hit("wire")).astype(np.int64)
        tt = np.asarray(sh.hit("time")).astype(np.int64) // 8
        key = ((evt * 3 + pl) * 1000 + w // 16) * 1000 + tt // 64
        toks.append(np.bincount(np.unique(key) // 3_000_000, minlength=sh.n))
    eff, pur = tp / tq, tp / max(tpp, 1)
    rep["baseline_T1_pandora_eff_pur_f1_q"] = [float(eff), float(pur), float(2 * eff * pur / (eff + pur))]
    rep["T1_nu_hit_fraction_visible_events"] = float(np.mean(nu_frac))
    toks = np.concatenate(toks)
    rep["tokens_16x64_pct_50_90_99_max"] = np.percentile(toks, [50, 90, 99]).tolist() + [int(toks.max())]
    rep["frac_events_over_4096_tokens"] = float((toks > 4096).mean())
    json.dump(rep, open(os.path.join(a.out, "audit.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

    # ---- figures ----
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax[0, 0].hist(ev["hit_cnt"], bins=100); ax[0, 0].set_xlabel("hits / event")
    ax[0, 1].hist(toks, bins=100); ax[0, 1].set_xlabel("charge tokens / event (16x64 cells)")
    ax[0, 2].hist(np.log10(1 + ntot), bins=100); ax[0, 2].set_xlabel("log10(1 + neutrino hits)"); ax[0, 2].set_yscale("log")
    m, y = lab["evt_nu"]
    ax[1, 0].hist([s_pnd[y[m] == 1], s_pnd[y[m] == 0]], bins=50, label=["visible ν", "no ν charge"], histtype="step", density=True)
    ax[1, 0].set_xlabel("Pandora max nu_score"); ax[1, 0].legend(fontsize=7)
    e = ev["nu_energy"][lab["energy"][0]]
    ax[1, 1].hist(e, bins=100, range=(0, 4)); ax[1, 1].set_xlabel("E_ν (GeV), FV CC visible")
    ax[1, 2].hist(dt, bins=100, range=(0.3, 0.7)); ax[1, 2].set_xlabel("beam-flash t − ν t (µs)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "audit.png"), dpi=120)

    # event display: first visible event with a nue/numu CC in FV
    sh = c.shards[0]
    sel = np.nonzero(event_labels({k: np.asarray(sh.ev(k)) for k in EV_KEYS})["energy"][0])[0]
    j = int(sel[0])
    h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
    o0, on = int(sh.ev("oph_off")[j]), int(sh.ev("oph_cnt")[j])
    pl = np.asarray(sh.hit("plane")[h0:h0 + hn]); w = np.asarray(sh.hit("wire")[h0:h0 + hn])
    tt = np.asarray(sh.hit("time")[h0:h0 + hn]) / 8; q = np.asarray(sh.hit("integral")[h0:h0 + hn]).astype(float)
    nu = np.asarray(sh.hit("nu_edep")[h0:h0 + hn]).astype(float) > HIT_NU_EDEP_MEV
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5), gridspec_kw=dict(width_ratios=[1, 1, 1.4, 1.2]))
    for p in range(3):
        mm = pl == p
        ax[p].scatter(w[mm & ~nu], tt[mm & ~nu], s=0.3, c="0.6")
        ax[p].scatter(w[mm & nu], tt[mm & nu], s=0.6, c="r")
        ax[p].set_title(f"plane {'UVY'[p]} (red: ν hits)"); ax[p].set_xlabel("wire"); ax[p].set_ylabel("tick")
    ch = np.asarray(sh.oph("ch")[o0:o0 + on]).astype(int); ot = np.asarray(sh.oph("t")[o0:o0 + on]); pe = np.asarray(sh.oph("pe")[o0:o0 + on]).astype(float).clip(0)
    ib = (ot > 3) & (ot < 6)
    pmt = np.bincount(ch[ib], weights=pe[ib], minlength=32)
    sc = ax[3].scatter(G.PMT_YZ[:, 1], G.PMT_YZ[:, 0], s=20 + 400 * pmt / max(pmt.max(), 1), c=np.log1p(pmt), cmap="viridis")
    vx = sh.ev("nu_vtx")[j]
    ax[3].scatter([vx[2]], [vx[1]], marker="*", c="r", s=120)
    ax[3].set_xlabel("z (cm)"); ax[3].set_ylabel("y (cm)"); ax[3].set_title("PMT PE in 3–6 µs (★ true vertex)")
    fig.suptitle(f"event {j}: pdg {int(sh.ev('nu_pdg')[j])} cc {int(sh.ev('is_cc')[j])} E={float(sh.ev('nu_energy')[j]):.2f} GeV")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "event_display.png"), dpi=120)


if __name__ == "__main__":
    main()
