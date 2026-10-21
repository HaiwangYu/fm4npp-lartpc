#!/usr/bin/env python
"""Per-hit nu probabilities of a trained hit-task head on a given list of events (for event displays).

Rebuilds the downstream model from RUN/metrics.json (its saved args), loads RUN/best.pt and predicts.
Usage: infer_hits.py RUN_DIR GIDX.npy OUT.npz
"""
import json
import sys

import numpy as np
import torch

from qlfm.data.dataset import Corpus, EventDataset, collate
from qlfm.ref import build_backbone
from qlfm.train_downstream import DownstreamModel, get_args, predict


def main():
    run, gfile, out = sys.argv[1:4]
    saved = json.load(open(f"{run}/metrics.json"))
    args = get_args(["--task", "hit", "--mode", "frozen", "--out", "/dev/null"])
    for k, v in saved.items():
        if hasattr(args, k):
            setattr(args, k, v)
    dev = torch.device("cuda")
    corpus = Corpus(args.corpus)
    gidx = np.load(gfile)
    backbone = build_backbone(args.backbone, args, args.fm)
    model = DownstreamModel(backbone, "hit", args.mode, bool(args.late_light)).to(dev)
    missing = model.load_state_dict(torch.load(f"{run}/best.pt", map_location=dev), strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith("backbone.") or "adapter_" in k], missing
    loader = torch.utils.data.DataLoader(EventDataset(corpus, gidx, truth=True), batch_size=4, shuffle=False,
                                         collate_fn=collate, num_workers=4)
    res = predict(model, loader, dev, "hit", lambda g: None)
    np.savez_compressed(out, **{k: v for k, v in res.items() if v is not None})
    print(run, "->", out, len(res["pred"]), "hits", flush=True)


if __name__ == "__main__":
    main()
