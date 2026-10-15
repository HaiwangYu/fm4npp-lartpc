"""Downstream training/evaluation: scratch | frozen | finetune | linear, one task per run (single GPU).

python -m qlfm.train_downstream --task evt_nu --mode frozen --fm RUN/fm.pt --label-frac 0.01 --out DIR
Writes best-on-val head/model, test predictions (preds_test.npz) and metrics (metrics.json).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.data.labels import EV_KEYS, HIT_NU_EDEP_MEV, N_OUT, event_labels
from qlfm.evaluate import metrics_for
from qlfm.models.fm import FM, FMCfg
from qlfm.models.heads import AttnPool, HitHead, MeanLinear
from qlfm.tokenize import TokCfg


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--task", required=True, choices=list(N_OUT))
    ap.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune", "linear"])
    ap.add_argument("--fm", default=None, help="pretrained fm.pt (frozen/finetune/linear)")
    ap.add_argument("--light", type=int, default=1, help="scratch only; otherwise taken from the FM")
    ap.add_argument("--label-frac", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--min-steps", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=50.0)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    # architecture (scratch); for pretrained models these come from the checkpoint
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pos-mode", default="centroid")
    ap.add_argument("--pos-freq", type=int, default=10)
    ap.add_argument("--cell-wires", type=int, default=16)
    ap.add_argument("--cell-ticks", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8192)
    return ap.parse_args(argv)


class DownstreamModel(nn.Module):
    def __init__(self, fm: FM, task, mode):
        super().__init__()
        self.fm, self.task, self.mode = fm, task, mode
        d = fm.cfg.d
        if task == "hit":
            self.head = HitHead(d, fm.charge.hit_dim)
        elif mode == "linear":
            self.head = MeanLinear(d, N_OUT[task])
        else:
            self.head = AttnPool(d, out_dim=N_OUT[task])

    def forward(self, b):
        if self.mode in ("frozen", "linear"):
            with torch.no_grad():
                o = self.fm(b)
        else:
            o = self.fm(b)
        if self.task == "hit":
            return self.head(o["h_hit"], o["zc"], o["ct"].hit2tok), o
        z = torch.cat([o["zc"], o["zl"]]) if o["zl"] is not None else o["zc"]
        evt = torch.cat([o["ct"].tok_evt, o["lt"].evt]) if o["zl"] is not None else o["ct"].tok_evt
        return self.head(z, evt, b["n_events"]), o


def loss_fn(task, out, b, y, class_w):
    if task == "hit":
        tgt = (b["hit_nu_edep"] > HIT_NU_EDEP_MEV).float()
        w = b["hit_q"].clamp(min=1.0).log1p()            # mild charge weighting
        return (F.binary_cross_entropy_with_logits(out, tgt, reduction="none") * w).sum() / w.sum()
    if task in ("evt_nu", "flavor"):
        return F.cross_entropy(out, y, weight=class_w)
    return F.huber_loss(out.squeeze(1), y.float(), delta=0.5)


def layer_groups(model, base_lr, decay):
    """Layer-wise lr decay for fine-tuning: head/last block get base_lr, embeddings the smallest."""
    depth = len(model.fm.encoder.blocks)
    groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head"):
            lid = depth + 1
        elif ".blocks." in n:
            lid = int(n.split(".blocks.")[1].split(".")[0]) + 1
        else:
            lid = 0
        groups.append({"params": [p], "lr_scale": decay ** (depth + 1 - lid),
                       "weight_decay": 0.05 if p.ndim >= 2 else 0.0})
    for g in groups:
        g["lr"] = base_lr * g["lr_scale"]
    return groups


@torch.no_grad()
def predict(model, loader, dev, task, labels_of):
    model.eval()
    preds, ys, gid, extra = [], [], [], {"hit_q": [], "hit_pnd": [], "hit_evt": [], "hit_tgt": []}
    for b in loader:
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(b)
        out = out.float()
        if task == "hit":
            preds.append(torch.sigmoid(out).cpu())
            extra["hit_q"].append(b["hit_q"].cpu())
            extra["hit_pnd"].append(b["hit_pnd_nu"].cpu())
            extra["hit_tgt"].append((b["hit_nu_edep"] > HIT_NU_EDEP_MEV).cpu())
            extra["hit_evt"].append(b["gidx"][b["hit_evt"]].cpu())
        else:
            preds.append(out.cpu())
        gid.append(b["gidx"].cpu())
    model.train()
    g = torch.cat(gid).numpy()
    res = {"gidx": g, "pred": torch.cat(preds).numpy()}
    if task == "hit":
        res.update({k: torch.cat(v).numpy() for k, v in extra.items()})
    else:
        res["y"] = labels_of(g)
    return res


def main(argv=None):
    args = get_args(argv)
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    corpus = Corpus(args.corpus)
    sp = np.load(args.splits)
    ev = {k: corpus.ev_array(k) for k in EV_KEYS}
    lab = event_labels(ev)
    elig, target = lab[args.task]
    labels_of = (lambda g: target[g]) if target is not None else (lambda g: None)

    pool = sp[f"lab_f{args.label_frac:g}"]
    tr = pool[elig[pool]]
    va = sp["val"][elig[sp["val"]]][: args.n_val]
    te = sp["test"][elig[sp["test"]]][: args.n_test]

    # ---- model ----
    if args.mode == "scratch":
        tok = TokCfg(cell_wires=args.cell_wires, cell_ticks=args.cell_ticks, max_tokens=args.max_tokens)
        fm = FM(FMCfg(d=args.d, depth=args.depth, heads=args.heads, use_light=bool(args.light), tok=tok,
                      pos_mode=args.pos_mode, pos_freq=args.pos_freq))
    else:
        s = torch.load(args.fm, map_location="cpu", weights_only=False)
        a = s["args"]
        tok = TokCfg(cell_wires=a["cell_wires"], cell_ticks=a["cell_ticks"], max_tokens=args.max_tokens)
        fm = FM(FMCfg(d=a["d"], depth=a["depth"], heads=a["heads"], use_light=bool(a["light"]), tok=tok,
                      pos_mode=a.get("pos_mode", "fourier"), pos_freq=a.get("pos_freq", 16)))
        fm.load_state_dict(s["fm"])
    model = DownstreamModel(fm, args.task, args.mode).to(dev)
    if args.mode in ("frozen", "linear"):
        for p in fm.parameters():
            p.requires_grad_(False)
        fm.eval()

    lr = args.lr or {"scratch": 3e-4, "finetune": 2e-4, "frozen": 1e-3, "linear": 3e-3}[args.mode]
    if args.mode == "finetune":
        opt = torch.optim.AdamW(layer_groups(model, lr, args.layer_decay), lr=lr)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
        for g in opt.param_groups:
            g["lr_scale"] = 1.0
    steps = int(np.clip(args.epochs * len(tr) / args.bs, args.min_steps, args.max_steps))
    class_w = None
    if args.task in ("evt_nu", "flavor"):
        cnt = np.bincount(target[tr], minlength=3).astype(np.float64)
        class_w = torch.tensor((cnt.sum() / np.maximum(cnt, 1) / 3) ** 0.5, dtype=torch.float32, device=dev)

    mk = lambda idx, shuffle, bs: torch.utils.data.DataLoader(
        EventDataset(corpus, idx, truth=(args.task == "hit")), batch_size=bs, shuffle=shuffle,
        collate_fn=collate, num_workers=args.workers, pin_memory=True, drop_last=shuffle,
        persistent_workers=shuffle)
    if len(tr) < 4:
        raise SystemExit(f"too few labeled training events ({len(tr)})")
    trl = mk(tr, True, min(args.bs, max(2, len(tr) // 2)))
    val = mk(va, False, args.bs)
    tel = mk(te, False, args.bs)
    yt = torch.from_numpy(target).to(dev) if target is not None else None
    info = dict(vars(args), n_train=len(tr), n_val=len(va), n_test=len(te), steps=steps,
                n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(json.dumps(info), flush=True)

    best, best_step, step, logs = -1e9, 0, 0, []
    ck = os.path.join(args.out, "best.pt")
    model.train()
    if args.mode in ("frozen", "linear"):
        fm.eval()
    done = False
    while not done:
        for b in trl:
            b = to_device(b, dev)
            for g in opt.param_groups:
                w = min(1.0, (step + 1) / max(1, steps // 20))
                g["lr"] = lr * g["lr_scale"] * w * 0.5 * (1 + np.cos(np.pi * step / steps))
            y = yt[b["gidx"]] if yt is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(b)
            loss = loss_fn(args.task, out.float(), b, y, class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == steps:
                m = metrics_for(args.task, predict(model, val, dev, args.task, labels_of))
                logs.append({"step": step, "loss": float(loss), **m})
                print(json.dumps(logs[-1]), flush=True)
                if m["score"] > best:
                    best, best_step = m["score"], step
                    torch.save({k: v for k, v in model.state_dict().items()
                                if args.mode not in ("frozen", "linear") or k.startswith("head")}, ck)
            if step >= steps or (time.time() - t0) / 60 > args.time_limit:
                done = True
                break
    sd = torch.load(ck, map_location=dev)
    model.load_state_dict(sd, strict=False)
    res = predict(model, tel, dev, args.task, labels_of)
    m = metrics_for(args.task, res)
    np.savez_compressed(os.path.join(args.out, "preds_test.npz"), **{k: v for k, v in res.items() if v is not None})
    out = dict(info, best_val=best, best_step=best_step, test=m, minutes=(time.time() - t0) / 60, val_log=logs)
    json.dump(out, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print("TEST", json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
