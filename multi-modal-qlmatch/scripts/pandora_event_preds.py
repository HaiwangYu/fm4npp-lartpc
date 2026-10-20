#!/usr/bin/env python
"""Per-event test predictions of the Pandora references (for event displays / easy-hard selection):
Pandora direct, and Pandora+BDT (+light) at 1 % labels, seed 0 -- same test subsets as the R2P runs.

Usage: pandora_event_preds.py OUT.npz
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pandora_refs import W, load_features, pick
from qlfm.data.dataset import Corpus
from qlfm.data.labels import EV_KEYS, event_labels


def main():
    c = Corpus(f"{W}/cache/bnb,{W}/cache/nue")
    sp = np.load(f"{W}/cache/splits_all.npz")
    X, names = load_features(c)
    col = {k: i for i, k in enumerate(names["charge"] + names["light"])}
    lab = event_labels({k: c.ev_array(k) for k in EV_KEYS})
    out = {}
    for task in ("evt_nu", "flavor", "energy"):
        elig, y = lab[task]
        te = pick(sp["test"][elig[sp["test"]]], 10000, 12)
        out[f"{task}_gidx"] = te
        # direct
        if task == "evt_nu":
            s = X[te, col["best_nu_score"]]
            out[f"{task}_direct"] = np.where(s > 0.5, 1, 0)             # no out-of-AV class
        elif task == "flavor":
            out[f"{task}_direct"] = np.where(X[te, col["best_is_numu_like"]] > 0, 0,
                                             np.where(X[te, col["best_is_nue_like"]] > 0, 1, 2))
        else:
            tr_all = sp["train"][elig[sp["train"]]]
            xq = X[tr_all, col["slice_logq_y"]]
            ok = xq > 0
            b, a0 = np.polyfit(xq[ok], y[tr_all][ok], 1)
            out[f"{task}_direct"] = a0 + b * X[te, col["slice_logq_y"]]
        # BDT (+light) at 1 %, seed 0
        pool = sp["lab_f0.01"]
        tr = pool[elig[pool]]
        kw = dict(max_iter=500, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=20, random_state=0)
        if task == "energy":
            out[f"{task}_bdt"] = HistGradientBoostingRegressor(**kw).fit(X[tr], y[tr]).predict(X[te])
        else:
            m = HistGradientBoostingClassifier(class_weight="balanced", **kw).fit(X[tr], y[tr])
            pr = np.zeros((len(te), 3))
            pr[:, m.classes_] = m.predict_proba(X[te])
            out[f"{task}_bdt"] = pr.argmax(1)
        out[f"{task}_y"] = y[te]
    np.savez(sys.argv[1], **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
