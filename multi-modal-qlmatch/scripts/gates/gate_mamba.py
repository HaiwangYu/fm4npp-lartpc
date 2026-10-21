"""Gate G: Mamba2 (mamba_ssm) forward/backward throughput for FM4NPP-m3-like stacks at long sequence lengths."""
import time
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm

d, L_layers = 256, 12
blocks = nn.ModuleList([nn.ModuleDict(dict(n=RMSNorm(d), m=Mamba2(d_model=d, d_state=16, d_conv=4, expand=2, headdim=64)))
                        for _ in range(L_layers)]).cuda()
head = nn.Linear(d, 90).cuda()
opt = torch.optim.AdamW(list(blocks.parameters()) + list(head.parameters()), 1e-4)
print("params %.2fM" % (sum(p.numel() for p in blocks.parameters()) / 1e6))
for L, B in [(2048, 32), (8192, 8), (16384, 4), (16384, 8)]:
    x0 = torch.randn(B, L, d, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(6):
        if i == 2:
            torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = x0
            for b in blocks:
                x = x + b["m"](b["n"](x))
            loss = head(x).float().pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); dt = (time.time() - t) / 4
    print(f"L={L:6d} B={B:3d}: {dt*1e3:7.1f} ms/step  {B*L/dt/1e6:6.2f} Mtok/s/GPU  mem {torch.cuda.max_memory_allocated()/1e9:5.1f} GB", flush=True)
