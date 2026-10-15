"""Multi-modal MAE pretraining (step-based, resumable, time-limited for 1 h debug jobs).

Launch: mpiexec -n NGPU --ppn 4 python -m qlfm.train_pretrain --out RUN_DIR [...]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from qlfm import dist
from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.models.fm import FM, MAE, FMCfg, MAECfg
from qlfm.tokenize import TokCfg


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--subset", default="train", help="key in the splits file (e.g. train, pre_f0.10)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--light", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32, help="events per GPU")
    ap.add_argument("--lr", type=float, default=1.5e-4, help="per 256 events")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pos-mode", default="centroid")
    ap.add_argument("--pos-freq", type=int, default=10)
    ap.add_argument("--cell-wires", type=int, default=16)
    ap.add_argument("--cell-ticks", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--light-weight", type=float, default=1.0)
    ap.add_argument("--strip-frac", type=float, default=0.5)
    ap.add_argument("--target-norm", type=int, default=1)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--time-limit", type=float, default=50.0, help="minutes before checkpoint+exit")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def build(args):
    tok = TokCfg(cell_wires=args.cell_wires, cell_ticks=args.cell_ticks, max_tokens=args.max_tokens)
    fm = FM(FMCfg(d=args.d, depth=args.depth, heads=args.heads, use_light=bool(args.light), tok=tok,
                  pos_mode=args.pos_mode, pos_freq=args.pos_freq))
    mae = MAE(fm, MAECfg(mask_ratio=args.mask_ratio, light_weight=args.light_weight, strip_frac=args.strip_frac,
                         target_norm=bool(args.target_norm)))
    return fm, mae


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def __iter__(self):
        g = np.random.default_rng(self.seed)
        while True:
            yield from g.permutation(self.n).tolist()


def lr_at(step, args, base):
    w = int(args.warmup * args.steps)
    if step < w:
        return base * (step + 1) / w
    p = (step - w) / max(1, args.steps - w)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def validate(mae, loader, dev, n_batches):
    mae.eval()
    state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(1234)                      # identical masks every validation
    acc, k = {}, 0
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mae(to_device(b, dev))
        for key, v in out.items():
            if key.startswith("loss"):
                acc[key] = acc.get(key, 0.0) + float(v)
        k += 1
    torch.cuda.set_rng_state(state)
    mae.train()
    res = {key: torch.tensor(v / max(k, 1), device=dev) for key, v in acc.items()}
    return {key: float(dist.all_reduce_mean(v)) for key, v in res.items()}


def main(argv=None):
    args = get_args(argv)
    t_start = time.time()
    rank, world, local = dist.init()
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed + rank)

    corpus = Corpus(args.corpus)
    sp = np.load(args.splits)
    train_idx, val_idx = sp[args.subset], sp["val"][: args.n_val]
    fm, mae = build(args)
    mae.to(dev)
    n_params = sum(p.numel() for p in fm.parameters())
    base_lr = args.lr * args.bs * world / 256
    decay = [p for n, p in mae.named_parameters() if p.ndim >= 2 and "pos" not in n]
    no_decay = [p for n, p in mae.named_parameters() if p.ndim < 2 or "pos" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=base_lr, betas=(0.9, 0.95))

    step = 0
    ck = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        mae.load_state_dict(s["mae"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
    if step >= args.steps:
        if dist.is_main():
            print(f"already done ({step} steps)", flush=True)
        return
    model = DDP(mae, device_ids=[dev.index], find_unused_parameters=True) if world > 1 else mae

    ds = EventDataset(corpus, train_idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, sampler=InfiniteSampler(len(ds), args.seed * 1000 + rank * 7 + step),
        collate_fn=collate, num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    vds = EventDataset(corpus, val_idx[rank::world])
    vloader = torch.utils.data.DataLoader(vds, batch_size=args.bs, collate_fn=collate, num_workers=2)
    n_val_batches = len(vds) // args.bs

    if dist.is_main():
        print(f"params(FM)={n_params/1e6:.2f}M world={world} train={len(train_idx)} lr={base_lr:.2e} "
              f"start_step={step}", flush=True)
        json.dump(vars(args) | {"n_params": n_params, "world": world, "n_train": len(train_idx)},
                  open(os.path.join(args.out, "args.json"), "w"), indent=1)
    logf = open(os.path.join(args.out, "log.jsonl"), "a") if dist.is_main() else None

    def save():
        if dist.is_main():
            torch.save({"mae": mae.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(args)},
                       ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            torch.save({"fm": fm.state_dict(), "args": vars(args), "step": step}, os.path.join(args.out, "fm.pt"))

    t_log, ev_log, acc = time.time(), 0, {}
    model.train()
    it = iter(loader)
    while step < args.steps:
        b = to_device(next(it), dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args, base_lr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        opt.step()
        step += 1
        ev_log += b["n_events"]
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        acc["gnorm"] = acc.get("gnorm", 0.0) + float(gn)
        acc["_n"] = acc.get("_n", 0) + 1

        if step % args.log_every == 0:
            dt = time.time() - t_log
            rec = {k: v / acc["_n"] for k, v in acc.items() if k != "_n"}
            rec.update(step=step, lr=opt.param_groups[0]["lr"], ev_per_s=ev_log * world / dt,
                       mem_gb=torch.cuda.max_memory_allocated() / 1e9, wall_min=(time.time() - t_start) / 60)
            if dist.is_main():
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            t_log, ev_log, acc = time.time(), 0, {}
        if step % args.val_every == 0 or step == args.steps:
            v = validate(mae, vloader, dev, n_val_batches)
            if dist.is_main():
                rec = {"step": step, **{"val_" + k: x for k, x in v.items()}}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
            save()
        # time limit: decided on rank 0, broadcast so all ranks stop together
        stop = torch.tensor(float((time.time() - t_start) / 60 > args.time_limit), device=dev)
        if world > 1:
            torch.distributed.broadcast(stop, 0)
        if stop.item():
            break
    save()
    if dist.is_main():
        print(f"stopped at step {step}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
