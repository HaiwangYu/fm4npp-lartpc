"""Task metrics. Each returns a dict with a 'score' used for model selection (higher is better)."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _eff_at_rej(y, s, rej):
    thr = np.quantile(s[y == 0], rej)
    return float((s[y == 1] > thr).mean())


def hit_metrics(r, thr=0.5):
    p, t, q = r["pred"], r["hit_tgt"].astype(bool), r["hit_q"].clip(min=0)
    sel = p > thr
    tp = (q * (sel & t)).sum()
    eff = tp / max((q * t).sum(), 1e-9)
    pur = tp / max((q * sel).sum(), 1e-9)
    f1 = 2 * eff * pur / max(eff + pur, 1e-9)
    rng = np.random.default_rng(0)
    k = rng.choice(len(p), size=min(len(p), 2_000_000), replace=False)
    auc = roc_auc_score(t[k], p[k]) if t[k].any() and (~t[k]).any() else float("nan")
    out = dict(score=float(f1), eff_q=float(eff), pur_q=float(pur), f1_q=float(f1), auc=float(auc))
    if "hit_pnd" in r:
        ps = r["hit_pnd"] > 0.5
        tp = (q * (ps & t)).sum()
        pe, pp = tp / max((q * t).sum(), 1e-9), tp / max((q * ps).sum(), 1e-9)
        out.update(pandora_eff_q=float(pe), pandora_pur_q=float(pp), pandora_f1_q=float(2 * pe * pp / max(pe + pp, 1e-9)))
    return out


def evt_nu_metrics(r):
    y, s = r["y"].astype(int), r["pred"].reshape(-1)
    auc = roc_auc_score(y, s)
    return dict(score=float(auc), auc=float(auc), eff_at_rej95=_eff_at_rej(y, s, 0.95),
                eff_at_rej99=_eff_at_rej(y, s, 0.99), n_pos=int(y.sum()), n_neg=int((1 - y).sum()))


def flavor_metrics(r):
    y, logits = r["y"].astype(int), r["pred"]
    pred = logits.argmax(1)
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (y, pred), 1)
    eff = cm.diagonal() / np.maximum(cm.sum(1), 1)
    pur = cm.diagonal() / np.maximum(cm.sum(0), 1)
    mf1 = f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)
    # nue-CC efficiency at 50% purity-free operating point: AUC of nue score
    pe = np.exp(logits - logits.max(1, keepdims=True))
    pe = pe / pe.sum(1, keepdims=True)
    auc_e = roc_auc_score(y == 1, pe[:, 1]) if (y == 1).any() and (y != 1).any() else float("nan")
    return dict(score=float(mf1), macro_f1=float(mf1), acc=float((pred == y).mean()),
                eff=eff.tolist(), pur=pur.tolist(), auc_nue=float(auc_e), confusion=cm.tolist())


def energy_metrics(r):
    y, p = r["y"], r["pred"].reshape(-1)
    rel = np.exp(p - y) - 1
    q16, q50, q84 = np.quantile(rel, [0.16, 0.5, 0.84])
    res = (q84 - q16) / 2
    return dict(score=float(-res), resolution=float(res), bias=float(q50),
                rms_log=float(np.sqrt(np.mean((p - y) ** 2))))


def metrics_for(task, r):
    return {"hit": hit_metrics, "evt_nu": evt_nu_metrics, "flavor": flavor_metrics,
            "energy": energy_metrics}[task](r)
