"""Per-event Pandora (open-data reco) and light-summary features for the round-2 reference baselines.

Writes <cache part>/pnd_feat.npy (float32 [n_events, F]) and pnd_feat.json (feature names).
The "best ν slice" is the slice with the highest nu_score (same definition as hit_pnd_nu in the cache).

Usage: python -m qlfm.data.pandora_features PART_DIR [PART_DIR ...]
"""
import json
import os
import sys
import time

import numpy as np

from qlfm import geometry as G
from qlfm.data.h5reader import OpenSampleFile

CHARGE_FEATS = [
    "n_slices", "n_nu_slices", "best_nu_score", "best_is_nue_like", "best_is_numu_like", "best_flashmatch",
    "vtx_x", "vtx_y", "vtx_z", "vtx_in_av",
    "n_pfp", "n_trk", "n_shw", "trkshr_max", "trkshr_min", "trkshr_mean",
    "slice_nhit_u", "slice_nhit_v", "slice_nhit_y", "slice_logq_u", "slice_logq_v", "slice_logq_y",
    "pfp_nhit_u", "pfp_nhit_v", "pfp_nhit_y", "pfp_logq_u", "pfp_logq_v", "pfp_logq_y",
    "slice_tspan_y", "slice_wspan_y", "slice_zc_y", "slice_t_mean",
]
LIGHT_FEATS = ["fl_n", "fl_logpe", "fl_t", "fl_zc", "fl_yc", "fl_zw", "fl_npmt", "dz_slice_flash"]


def _best_rows(evt, score, n):
    """Index of the max-score row per event (-1 if none)."""
    best = np.full(n, -1, np.int64)
    if len(evt) == 0:
        return best
    order = np.lexsort((-score, evt))
    first = np.ones(len(order), bool)
    first[1:] = evt[order][1:] != evt[order][:-1]
    best[evt[order][first]] = order[first]
    return best


def features_for_part(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    lo, hi = meta["ev_lo"], meta["ev_hi"]
    n = hi - lo
    f = OpenSampleFile(meta["source"])
    rd = lambda t, cols: (lambda r: (r[0], r[1] - lo))(f.read_range(t, lo, hi, cols))
    X = np.zeros((n, len(CHARGE_FEATS) + len(LIGHT_FEATS)), np.float32)
    col = {k: i for i, k in enumerate(CHARGE_FEATS + LIGHT_FEATS)}

    # ---- slices (pandoraPrimary) ----
    pp, pe = rd("pandoraPrimary_table", ["slice_id", "slice_pdg", "nu_score", "flashmatch_score", "vtx"])
    X[:, col["n_slices"]] = np.bincount(pe, minlength=n)
    isnu = np.isin(pp["slice_pdg"], (12, 14))
    X[:, col["n_nu_slices"]] = np.bincount(pe[isnu], minlength=n)
    b = _best_rows(pe, np.where(pp["nu_score"] > -998, pp["nu_score"], -999.0), n)
    ok = b >= 0
    X[:, col["best_nu_score"]] = np.where(ok, pp["nu_score"][np.maximum(b, 0)], -1.0).clip(-1, 1)
    pdg = np.where(ok, pp["slice_pdg"][np.maximum(b, 0)], 0)
    X[:, col["best_is_nue_like"]] = pdg == 12
    X[:, col["best_is_numu_like"]] = pdg == 14
    fm = np.where(ok, pp["flashmatch_score"][np.maximum(b, 0)], 9999.0)
    X[:, col["best_flashmatch"]] = np.log1p(np.clip(fm, 0, 9999))
    vtx = np.where(ok[:, None], pp["vtx"][np.maximum(b, 0)], -999.0)
    X[:, [col["vtx_x"], col["vtx_y"], col["vtx_z"]]] = vtx
    X[:, col["vtx_in_av"]] = G.in_volume(vtx) & ok

    # ---- ν-slice PFPs ----
    pf, pfe = rd("pandoraPfp_table", ["pfp_pdg", "trkshr_score"])
    X[:, col["n_pfp"]] = np.bincount(pfe, minlength=n)
    X[:, col["n_trk"]] = np.bincount(pfe[pf["pfp_pdg"] == 13], minlength=n)
    X[:, col["n_shw"]] = np.bincount(pfe[pf["pfp_pdg"] == 11], minlength=n)
    ts = pf["trkshr_score"].astype(np.float64)
    tmax = np.full(n, -1.0); tmin = np.full(n, 2.0); tsum = np.zeros(n)
    np.maximum.at(tmax, pfe, ts); np.minimum.at(tmin, pfe, ts); np.add.at(tsum, pfe, ts)
    has = X[:, col["n_pfp"]] > 0
    X[:, col["trkshr_max"]] = np.where(has, tmax, -1)
    X[:, col["trkshr_min"]] = np.where(has, tmin, -1)
    X[:, col["trkshr_mean"]] = np.where(has, tsum / np.maximum(X[:, col["n_pfp"]], 1), -1)

    # ---- hits: best-slice and ν-PFP membership (cache columns + pandoraHit.pfp_id) ----
    L = lambda k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r")
    cnt = np.asarray(L("ev_hit_cnt"))
    evt = np.repeat(np.arange(n), cnt)
    plane = np.asarray(L("hit_plane")).astype(np.int64)
    q = np.asarray(L("hit_integral")).astype(np.float64).clip(0)
    t = np.asarray(L("hit_time")).astype(np.float64) / 8.0
    w = np.asarray(L("hit_wire")).astype(np.float64)
    in_slice = np.asarray(L("hit_pnd_nu")).astype(bool)
    ph, phe = rd("pandoraHit_table", ["hit_id", "pfp_id"])
    hs = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    in_pfp = np.zeros(len(q), bool)
    in_pfp[hs[phe] + ph["hit_id"]] = ph["pfp_id"] >= 0
    for p, s in enumerate("uvy"):
        m = in_slice & (plane == p)
        X[:, col[f"slice_nhit_{s}"]] = np.bincount(evt[m], minlength=n)
        X[:, col[f"slice_logq_{s}"]] = np.log1p(np.bincount(evt[m], weights=q[m], minlength=n))
        m = in_pfp & (plane == p)
        X[:, col[f"pfp_nhit_{s}"]] = np.bincount(evt[m], minlength=n)
        X[:, col[f"pfp_logq_{s}"]] = np.log1p(np.bincount(evt[m], weights=q[m], minlength=n))
    m = in_slice & (plane == 2)
    big = 1e9
    tmx = np.full(n, -big); tmn = np.full(n, big); wmx = np.full(n, -big); wmn = np.full(n, big)
    np.maximum.at(tmx, evt[m], t[m]); np.minimum.at(tmn, evt[m], t[m])
    np.maximum.at(wmx, evt[m], w[m]); np.minimum.at(wmn, evt[m], w[m])
    hasy = X[:, col["slice_nhit_y"]] > 0
    X[:, col["slice_tspan_y"]] = np.where(hasy, tmx - tmn, 0)
    X[:, col["slice_wspan_y"]] = np.where(hasy, wmx - wmn, 0)
    qs = np.bincount(evt[m], weights=q[m], minlength=n)
    zc = np.bincount(evt[m], weights=q[m] * (0.25 + 0.3 * w[m]), minlength=n) / np.maximum(qs, 1e-9)
    X[:, col["slice_zc_y"]] = np.where(hasy, zc, -1)
    ms = in_slice
    qa = np.bincount(evt[ms], weights=q[ms], minlength=n)
    X[:, col["slice_t_mean"]] = np.where(qa > 0, np.bincount(evt[ms], weights=q[ms] * t[ms], minlength=n) / np.maximum(qa, 1e-9), -1)

    # ---- light summary (largest beam flash, from the cache) ----
    fl_pmt = np.asarray(L("ev_fl_pmt")).astype(np.float64)
    pe = fl_pmt.sum(1)
    X[:, col["fl_n"]] = np.asarray(L("ev_fl_n"))
    X[:, col["fl_logpe"]] = np.log1p(np.asarray(L("ev_fl_pe")).clip(0))
    X[:, col["fl_t"]] = np.asarray(L("ev_fl_t"))
    zc_f = (fl_pmt @ G.PMT_YZ[:, 1]) / np.maximum(pe, 1e-9)
    yc_f = (fl_pmt @ G.PMT_YZ[:, 0]) / np.maximum(pe, 1e-9)
    zw_f = np.sqrt(np.maximum((fl_pmt @ G.PMT_YZ[:, 1] ** 2) / np.maximum(pe, 1e-9) - zc_f ** 2, 0))
    hasf = pe > 0
    X[:, col["fl_zc"]] = np.where(hasf, zc_f, -1)
    X[:, col["fl_yc"]] = np.where(hasf, yc_f, -999)
    X[:, col["fl_zw"]] = np.where(hasf, zw_f, -1)
    X[:, col["fl_npmt"]] = (fl_pmt > 0).sum(1)
    X[:, col["dz_slice_flash"]] = np.where(hasf & hasy, X[:, col["slice_zc_y"]] - zc_f, 2000)
    f.close()
    np.save(os.path.join(d, "pnd_feat.npy"), X)
    json.dump({"charge": CHARGE_FEATS, "light": LIGHT_FEATS}, open(os.path.join(d, "pnd_feat.json"), "w"))
    return n


if __name__ == "__main__":
    for d in sys.argv[1:]:
        t0 = time.time()
        n = features_for_part(d)
        print(f"{d}: {n} events, {time.time() - t0:.0f}s", flush=True)
