"""Gate G: pimm PT-v3m2 (Panda backbone, ~21M variant) forward/backward on synthetic planar point clouds."""
import time
import numpy as np
import torch
import pimm.models  # noqa: F401  (registers models)
from pimm.models.builder import build_model

cfg = dict(type="PT-v3m2", in_channels=4, order=("hilbert", "hilbert-trans", "z", "z-trans"), stride=(2, 2, 2, 2),
           enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 384), enc_num_head=(2, 4, 8, 16, 24),
           enc_patch_size=(256,) * 5, mlp_ratio=4, qkv_bias=True, qk_scale=None, layer_scale=1e-5, attn_drop=0.0,
           proj_drop=0.0, drop_path=0.3, shuffle_orders=True, pre_norm=True, enable_rpe=False, enable_flash=True,
           upcast_attention=False, upcast_softmax=False, traceable=True, enc_mode=True, mask_token=True,
           cpe_first_layer_only=False, cpe_shared_weight=False)
m = build_model(cfg).cuda()
print("params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
opt = torch.optim.AdamW(m.parameters(), 1e-4)
grid = 0.00055
rng = np.random.default_rng(0)
for B, npts in [(16, 5000), (48, 5000)]:
    coords, feats, off = [], [], []
    for b in range(B):
        c = np.zeros((npts, 3), np.float32); c[:, :2] = rng.uniform(-0.9, 0.9, (npts, 2))
        coords.append(c); feats.append(np.concatenate([c, rng.uniform(-1, 1, (npts, 1)).astype(np.float32)], 1))
        off.append((b + 1) * npts)
    coord = torch.from_numpy(np.concatenate(coords)).cuda()
    data = dict(coord=coord, grid_coord=torch.div(coord - coord.min(0).values, grid, rounding_mode="floor").int(),
                feat=torch.from_numpy(np.concatenate(feats)).cuda(), offset=torch.tensor(off).cuda(), grid_size=grid)
    torch.cuda.reset_peak_memory_stats()
    for i in range(5):
        if i == 2:
            torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(dict(data))
            loss = out.feat.float().pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); dt = (time.time() - t) / 3
    print(f"B={B} x {npts} pts: {dt*1e3:.0f} ms/step, {B/dt:.0f} clouds/s/GPU, mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
