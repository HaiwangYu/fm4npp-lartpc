"""Round 4 T5 zero-shot point: the pretrained multi-modal MAE (charge+light, qlfm CL) predicts the light of every
event from the charge alone through its own decoder (all charge tokens visible, all 32 PMT tokens masked), with no
labelled fine-tuning. Same test subset, outputs and metrics as `train_downstream --task light`.

python -m qlfm.t5_zeroshot --ckpt PRE/ckpt.pt --out DIR
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from qlfm.data.dataset import Corpus, EventDataset, collate, to_device
from qlfm.evaluate import metrics_for
from qlfm.models.encoder import pack
from qlfm.models.fm import FM, MAE, FMCfg, MAECfg
from qlfm.tokenize import TokCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=os.path.expandvars("$QLFM_WORK/cache/bnb,$QLFM_WORK/cache/nue"))
    ap.add_argument("--splits", default=os.path.expandvars("$QLFM_WORK/cache/splits_all.npz"))
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--time-limit", type=float, default=50.0)        # accepted for run_queue.sh; unused
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = s["args"]
    assert a["light"], "zero-shot charge->light needs a charge+light pretrained MAE"
    tok = TokCfg(cell_wires=a["cell_wires"], cell_ticks=a["cell_ticks"], max_tokens=a["max_tokens"])
    fm = FM(FMCfg(d=a["d"], depth=a["depth"], heads=a["heads"], use_light=True, tok=tok,
                  pos_mode=a.get("pos_mode", "fourier"), pos_freq=a.get("pos_freq", 16)))
    mae = MAE(fm, MAECfg(mask_ratio=a["mask_ratio"], strip_frac=a["strip_frac"], light_weight=a["light_weight"],
                         target_norm=bool(a["target_norm"])))
    mae.load_state_dict(s["mae"])
    dev = torch.device("cuda")
    mae = mae.to(dev).eval()

    c = Corpus(args.corpus)
    sp = np.load(args.splits)
    te = np.sort(np.random.default_rng(12).permutation(sp["test"])[:args.n_test])   # = train_downstream's test pick
    dl = torch.utils.data.DataLoader(EventDataset(c, te), batch_size=args.bs, shuffle=False, collate_fn=collate,
                                     num_workers=6, pin_memory=True)
    preds, tgts, gid = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in dl:
            b = to_device(b, dev)
            n = b["n_events"]
            ct, lt = fm.tokenize(b)
            xc, xl, _ = fm.embed(ct, lt)
            (zc, _), _ = fm.encode(xc, ct.tok_evt, None, None, n)                  # charge only, all tokens visible
            dc = mae.enc2dec(zc) + mae.dec_pos_c(fm.charge.pos_embed(ct))
            dlt = mae.mask_l.to(dc.dtype).expand(xl.shape[0], -1) + mae.dec_pos_l(fm.light.pos_embed(lt))
            perm, cu, maxlen, _ = pack([ct.tok_evt, lt.evt], n)
            y = mae.decoder(torch.cat([dc, dlt.to(dc.dtype)])[perm], cu, maxlen)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.numel(), device=dev)
            yl = y[inv[dc.shape[0]:]]
            preds.append(mae.head_l(yl).float().view(n, -1, lt.feat.shape[1]).half().cpu())
            tgts.append(lt.feat.view(n, -1, lt.feat.shape[1]).half().cpu())
            gid.append(b["gidx"].cpu())
    res = {"gidx": torch.cat(gid).numpy(), "pred": torch.cat(preds).numpy(), "light_tgt": torch.cat(tgts).numpy()}
    m = metrics_for("light", res)
    np.savez_compressed(os.path.join(args.out, "preds_test.npz"), **res)
    json.dump(dict(vars(args), n_test=len(te), test=m, minutes=(time.time() - t0) / 60, mode="zeroshot"),
              open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print("TEST", json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
