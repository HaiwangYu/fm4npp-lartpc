#!/usr/bin/env python
"""Diagnose the charge-MAE objective: component losses vs mean baseline; single-batch overfit test."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.train_pretrain import build, get_args


def components(mae, b):
    """Return per-component masked-charge MSE for model and for a per-dim mean predictor."""
    fm, n = mae.fm, b["n_events"]
    ct, lt = fm.tokenize(b)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    out = {}
    # re-run the forward but keep predictions: copy of MAE.forward essentials
    import qlfm.models.fm as M
    captured = {}
    orig = F.mse_loss

    def spy(pred, tgt, *a, **k):
        captured.setdefault("pairs", []).append((pred.detach(), tgt.detach()))
        return orig(pred, tgt, *a, **k)

    M.F.mse_loss = spy
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mae(b)
    M.F.mse_loss = orig
    pc, tc = captured["pairs"][0]
    ns = pc.shape[1] - 2
    mean = tc.mean(0, keepdim=True)
    for name, sl in (("subgrid", slice(0, ns)), ("stats", slice(ns, ns + 2)), ("all", slice(None))):
        out[name] = dict(model=float(((pc[:, sl] - tc[:, sl]) ** 2).mean()),
                         mean_pred=float(((mean[:, sl] - tc[:, sl]) ** 2).mean()),
                         pred_std=float(pc[:, sl].std(0).mean()), tgt_std=float(tc[:, sl].std(0).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--extra", default="")
    ap.add_argument("--train", type=int, default=0, help="1: train on a stream of events, eval held-out")
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    c = Corpus(a.corpus)
    ds = EventDataset(c, np.arange(32))
    b = to_device(collate([ds[i] for i in range(32)]), "cuda")
    args = get_args(["--out", "/tmp/x"] + a.extra.split())
    fm, mae = build(args)
    mae.cuda()
    if a.ckpt:
        s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        mae.load_state_dict(s["mae"])
        print("ckpt components:", json.dumps(components(mae, b), indent=1))
        return
    opt = torch.optim.AdamW(mae.parameters(), lr=a.lr, weight_decay=0.05)
    if a.train:
        tl = torch.utils.data.DataLoader(EventDataset(c, np.arange(0, 40000)), batch_size=32, shuffle=True,
                                         collate_fn=collate, num_workers=6, drop_last=True)
        it, i = iter(tl), 0
        while i < a.steps:
            try:
                bb = to_device(next(it), "cuda")
            except StopIteration:
                it = iter(tl)
                continue
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (i + 1) / 100)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = mae(bb)
            opt.zero_grad()
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            opt.step()
            if i % 200 == 0:
                print(i, {k: round(float(v), 4) for k, v in o.items() if k.startswith("loss")}, flush=True)
            i += 1
        hv = EventDataset(c, np.arange(70000, 70064))
        hb = to_device(collate([hv[j] for j in range(64)]), "cuda")
        mae.eval()
        print("heldout components:", json.dumps(components(mae, hb)))
        return
    for i in range(a.steps):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)          # identical masks every step -> pure overfit test
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = mae(b)
        opt.zero_grad()
        o["loss"].backward()
        opt.step()
        if i % 50 == 0 or i == a.steps - 1:
            print(i, {k: round(float(v), 4) for k, v in o.items() if k.startswith("loss")}, flush=True)
    print("overfit components:", json.dumps(components(mae, b), indent=1))


if __name__ == "__main__":
    main()
