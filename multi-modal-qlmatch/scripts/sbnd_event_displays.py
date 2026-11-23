#!/usr/bin/env python
"""SBND event displays (issue #14): random events from the simulation test list, beam-on and off-beam data (held-out
test files of the real-data fine-tune split), with the hit and light predictions of several models.

One page per event (4 x 3 panels):
  row 0  charge (global channel vs raw tick, colour = log10 hit integral) | measured PMT light | event info
  row 1  references: truth nu hits (simulation) | Wire-Cell nu footprint | footprint of candidates passing the PR
         cosmic tagger
  row 2  hit p(nu) of each model
  row 3  each model's charge-only light prediction (T5 head; filled = predicted, black ring = measured)
Channels: TPC 0 U, V, Y then TPC 1 U, V, Y; raw tick = the TDC tick of the hit (TPC 1 un-mirrored).
PMT views: total PE per PMT in -3..21 us (beam clock) at the PMT's (z, y); TPC 0 PMTs left, TPC 1 PMTs right.

Usage (QLFM_DETECTOR=sbnd, on a GPU node):
  sbnd_event_displays.py OUT_DIR LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--n 20] [--seed 0]
  -> OUT_DIR/{sim,on,off}_NN.png and OUT_DIR/displays.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from qlfm import geometry as G
from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.data.labels import EV_KEYS, HIT_NU_EDEP_MEV, event_labels
from qlfm.eval_perturb import load_run
from qlfm.train_downstream import light_target

W = os.environ.get("QLFM_WORK", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work")
SIM = f"{W}/cache/sbnd_m/bnb,{W}/cache/sbnd_m/nuecc,{W}/cache/sbnd_m/ncdr"
DATA = f"{W}/cache/sbnd_data_m/bnb-opendata,{W}/cache/sbnd_data_m/offbeam-intime-run1"
NW = np.array(G.N_WIRES)
CH0 = np.concatenate([[0], np.cumsum(NW)[:-1]])
NCH = int(NW.sum())
INK, MUTED = "#2b2b2a", "#8a8980"


def pmt_xyz():
    d = json.load(open(os.environ.get("QLFM_DETECTOR_JSON", os.path.join(
        os.path.dirname(__file__), "..", "sbnd-h5maker", "sbnd_detector.json"))))
    p = sorted((o for o in d["opdets"] if o["type"].startswith("pmt")), key=lambda o: o["channel"])
    return np.array([[o["x"], o["y"], o["z"]] for o in p])


PXYZ = pmt_xyz()
assert len(PXYZ) == G.N_PMT


def pick_events(n, seed):
    rng = np.random.default_rng(seed)
    sim_ev = np.load(f"{W}/sbnd_data/sim_eval_events.npy")
    test = np.load(f"{W}/cache/splits_sbnd_data.npz")["test"]
    dc = Corpus(DATA)
    smp = dc.sample_id()[test]
    return {"sim": np.sort(rng.choice(sim_ev, n, replace=False)),
            "on": np.sort(rng.choice(test[smp == 0], n, replace=False)),
            "off": np.sort(rng.choice(test[smp == 1], n, replace=False))}


@torch.no_grad()
def infer(run, corpus, idx, dev):
    a, ns, model, _ = load_run(run, dev)
    model.width = "native"
    model.tasks = [t for t in ("hit", "evt_nu", "light") if t in a["tasks"]]
    model.light_off_pass = "light" in model.tasks and bool(getattr(model.backbone, "light_on", False))
    model.eval()
    dl = torch.utils.data.DataLoader(EventDataset(corpus, idx, truth=True), batch_size=4, shuffle=False,
                                     collate_fn=collate, num_workers=2)
    res = {}
    torch.manual_seed(0)
    for b in dl:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b)
        he = b["hit_evt"].cpu().numpy()
        ph = torch.sigmoid(o["hit"].float()).reshape(-1).cpu().numpy()
        lg = o["evt_nu"].float().cpu().numpy()
        lp = o["light"].float().cpu().numpy() if "light" in o else None
        lt = light_target(b).float().cpu().numpy()
        for i, g in enumerate(b["gidx"].cpu().numpy()):
            res[int(g)] = dict(p_hit=ph[he == i], logodds=float(lg[i, 1] - np.logaddexp(lg[i, 0], lg[i, 2])),
                               light=None if lp is None else np.expm1(lp[i, :, -1]).clip(min=0),
                               light_meas=np.expm1(lt[i, :, -1]))
    return res


def load_event(c, g):
    s, j = c.locate(g)
    sh = c.shards[s]
    h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
    e = {k: np.asarray(sh.hit(k)[h0:h0 + hn]) for k in ("plane", "wire", "time", "integral", "tpc", "nu_edep",
                                                        "pnd_nu", "pnd_nu_ct0")}
    vt = e["time"].astype(np.float32) / 8
    e["tick"] = np.where(e["tpc"] == 0, vt, G.TPC1_FLIP_TICK - vt)
    e["ch"] = e["tpc"].astype(np.int64) * NCH + CH0[e["plane"]] + e["wire"].astype(np.int64)
    e["q"] = e["integral"].astype(np.float32).clip(min=1)
    e.update({k: np.asarray(sh.ev(k)[j]) for k in ("run", "subrun", "event", "fl_t", "fl_pe")})
    for k in ("nu_pdg", "is_cc", "nu_energy", "nu_vtx"):
        e[k] = np.asarray(sh.ev(k)[j])
    return e


def frame(ax, title):
    for t in range(2):
        for p in range(3):
            x = t * NCH + CH0[p]
            ax.axvline(x, color=MUTED if p else INK, lw=0.4 if p else 0.8)
            ax.text(x + NW[p] / 2, 1.0, f"TPC{t} {'UVY'[p]}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=5, color=MUTED)
    ax.set_xlim(0, 2 * NCH)
    ax.set_ylim(0, 3427)                                   # readout window (ticks)
    ax.set_title(title, fontsize=7, pad=9)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("channel", fontsize=5, labelpad=1)
    ax.set_ylabel("tick", fontsize=5, labelpad=1)


def hits(ax, e, c, cmap, vmin, vmax, order=None, s=0.5):
    o = np.argsort(c) if order is None else order
    return ax.scatter(e["ch"][o], e["tick"][o], c=c[o], s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
                      rasterized=True)


def label_panel(ax, e, m, title, color):
    ax.scatter(e["ch"], e["tick"], s=0.3, c="#d6d5ce", linewidths=0, rasterized=True)
    ax.scatter(e["ch"][m], e["tick"][m], s=0.8, c=color, linewidths=0, rasterized=True)
    frame(ax, title)


def pmt_panel(ax, pe, ref, scale, title, color):
    z = PXYZ[:, 2] + np.where(PXYZ[:, 0] > 0, 560.0, 0.0)
    y = PXYZ[:, 1]
    ax.scatter(z, y, s=4, c="#e9e8e2", linewidths=0)
    if pe is not None:
        ax.scatter(z, y, s=scale * pe, c=color, alpha=0.75, linewidths=0)
    if ref is not None:
        ax.scatter(z, y, s=scale * ref, facecolors="none", edgecolors=INK, linewidths=0.5)
    ax.axvline(530, color=MUTED, lw=0.6)
    ax.text(250, 1.0, "TPC 0 PMTs", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5,
            color=MUTED)
    ax.text(810, 1.0, "TPC 1 PMTs", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5,
            color=MUTED)
    ax.set_xlim(-20, 1080)
    ax.set_ylim(-230, 230)
    ax.set_xticks([])
    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=7, pad=9)


def f1(sel, ref, q):
    tp, tt, ss = (q * (sel & ref)).sum(), (q * ref).sum(), (q * sel).sum()
    if tt == 0 and ss == 0:
        return None
    return float(2 * tp / max(tt + ss, 1e-9))


def render(kind, k, g, e, preds, names, out):
    fig = plt.figure(figsize=(15, 12.5))
    gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.18, left=0.04, right=0.99, top=0.95, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0])
    hits(ax, e, np.log10(e["q"]), "viridis", 1.5, 3.5)
    frame(ax, "Charge (colour = log10 hit integral)")
    meas = preds[names[0]]["light_meas"]
    scale = 300.0 / max(meas.max(), max((p["light"].max() for p in preds.values() if p["light"] is not None),
                                        default=1), 1)
    pmt_panel(fig.add_subplot(gs[0, 1]), meas, None, scale, f"Measured light: {meas.sum():.0f} PE (-3..21 µs)", "0.4")
    truth = e["nu_edep"].astype(np.float32) > HIT_NU_EDEP_MEV
    wc, wc0 = e["pnd_nu"] > 0, e["pnd_nu_ct0"] > 0
    q = e["q"].astype(np.float64)
    head = {"sim": "simulation (BNB CV / νe CC / NC)", "on": "beam-on data", "off": "off-beam data"}[kind]
    txt = [f"{head}", f"run {int(e['run'])}  subrun {int(e['subrun'])}  event {int(e['event'])}",
           f"{len(q):,} hits;  brightest flash {float(e['fl_pe']):.0f} PE at {float(e['fl_t']):.2f} µs",
           f"Wire-Cell ν hits: {int(wc.sum()):,} (after cosmic tagger {int(wc0.sum()):,})"]
    if kind == "sim":
        txt.append(f"truth: ν pdg {int(e['nu_pdg'])}, {'CC' if int(e['is_cc']) == 1 else 'NC'}, "
                   f"E = {float(e['nu_energy']):.2f} GeV, ν hits {int(truth.sum()):,}")
    txt.append("")
    summ = {}
    for n in names:
        p = preds[n]
        sel = p["p_hit"] > 0.5
        r = dict(p_nu=float(1 / (1 + np.exp(-p["logodds"]))), n_nu_hits=int(sel.sum()), f1_wc=f1(sel, wc, q),
                 f1_truth=f1(sel, truth, q) if kind == "sim" else None,
                 pe_pred=None if p["light"] is None else float(p["light"].sum()))
        summ[n] = r
        fmt = lambda v: "--" if v is None else f"{v:.2f}"
        txt.append(f"{n}: p(ν) {r['p_nu']:.2f};  F1 vs WC {fmt(r['f1_wc'])}"
                   + (f", vs truth {fmt(r['f1_truth'])}" if kind == "sim" else "")
                   + ("" if r["pe_pred"] is None else f";  pred. light {r['pe_pred']:.0f} PE"))
    axt = fig.add_subplot(gs[0, 2])
    axt.axis("off")
    axt.text(0, 1, "\n".join(txt), va="top", fontsize=7.5, color=INK, family="DejaVu Sans")
    ax = fig.add_subplot(gs[1, 0])
    if kind == "sim":
        label_panel(ax, e, truth, f"Truth ν hits (> {HIT_NU_EDEP_MEV} MeV)", "#e34948")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "real data: no truth", ha="center", va="center", fontsize=9, color=MUTED)
    label_panel(fig.add_subplot(gs[1, 1]), e, wc, "Wire-Cell ν-candidate footprint", "#2a78d6")
    label_panel(fig.add_subplot(gs[1, 2]), e, wc0, "… candidates passing the PR cosmic tagger", "#6250d6")
    for i, n in enumerate(names[:3]):
        ax = fig.add_subplot(gs[2, i])
        sc = hits(ax, e, preds[n]["p_hit"], "coolwarm", 0, 1)
        frame(ax, f"{n}: hit p(ν)")
        if i == 2:
            cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
            cb.ax.tick_params(labelsize=5)
        lp = preds[n]["light"]
        pmt_panel(fig.add_subplot(gs[3, i]), lp, meas, scale,
                  f"{n}: light from charge" + ("" if lp is None else f", {lp.sum():.0f} PE"), "#eb6834")
    fig.suptitle(f"{head}: event {k + 1}", fontsize=10, x=0.04, ha="left")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("models", nargs="+", help="LABEL=RUN_DIR")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert G.DETECTOR == "sbnd"
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    models = [m.split("=", 1) for m in a.models]
    names = [n for n, _ in models]
    ev = pick_events(a.n, a.seed)
    corp = {"sim": Corpus(SIM), "on": Corpus(DATA), "off": Corpus(DATA)}
    allp = {kind: {} for kind in ev}
    for n, run in models:
        for kind, idx in ev.items():
            for g, r in infer(run, corp[kind], idx, dev).items():
                allp[kind].setdefault(g, {})[n] = r
        print("inferred", n, flush=True)
    summary = {}
    for kind, idx in ev.items():
        for k, g in enumerate(idx):
            e = load_event(corp[kind], int(g))
            summary[f"{kind}_{k:02d}"] = dict(gidx=int(g), run=int(e["run"]), subrun=int(e["subrun"]),
                                             event=int(e["event"]),
                                             models=render(kind, k, int(g), e, allp[kind][int(g)], names,
                                                           os.path.join(a.out, f"{kind}_{k:02d}.png")))
        print("rendered", kind, flush=True)
    json.dump(dict(models=dict(models), seed=a.seed, events={k: v.tolist() for k, v in ev.items()}, pages=summary),
              open(os.path.join(a.out, "displays.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
