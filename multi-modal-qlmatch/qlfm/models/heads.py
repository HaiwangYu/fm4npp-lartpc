"""Small downstream heads on top of (frozen or trainable) FM token features."""
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean

from qlfm.models.encoder import mlp


class AttnPool(nn.Module):
    """Attentive probe: n_q learned queries cross-attend over all tokens of each event."""

    def __init__(self, d, n_q=4, heads=8, out_dim=1, hidden=256):
        super().__init__()
        self.n_q, self.h, self.dh = n_q, heads, d // heads
        self.norm = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(n_q, heads, self.dh) * 0.02)
        self.kv = nn.Linear(d, 2 * d)
        self.out = mlp([n_q * d, hidden, out_dim])

    def forward(self, z, evt, n_events):
        n, d = z.shape
        k, v = self.kv(self.norm(z)).float().view(n, 2, self.h, self.dh).unbind(1)
        s = torch.einsum("nhd,qhd->nqh", k, self.q.float()) / self.dh ** 0.5      # [N, Q, H]
        smax = scatter_max(s, evt, dim=0, dim_size=n_events)[0]
        e = torch.exp(s - smax[evt])
        a = e / scatter_add(e, evt, dim=0, dim_size=n_events)[evt].clamp(min=1e-20)
        pooled = scatter_add(a[..., None] * v[:, None], evt, dim=0, dim_size=n_events)  # [B, Q, H, Dh]
        return self.out(pooled.reshape(n_events, self.n_q * d))


class MeanLinear(nn.Module):
    """Linear probe on mean-pooled token features (diagnostic)."""

    def __init__(self, d, out_dim=1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin = nn.Linear(d, out_dim)

    def forward(self, z, evt, n_events):
        return self.lin(self.norm(scatter_mean(z.float(), evt, dim=0, dim_size=n_events)))


class HitHead(nn.Module):
    """Per-hit logit from the hit embedding and its token's contextual feature."""

    def __init__(self, d, hit_dim=128, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = mlp([hit_dim + d, hidden, hidden, 1])

    def forward(self, h_hit, z_tok, hit2tok):
        m = hit2tok >= 0
        zt = self.norm(z_tok.float())[hit2tok.clamp(min=0)]
        logit = self.net(torch.cat([h_hit.float(), zt], 1)).squeeze(1)
        return torch.where(m, logit, torch.full_like(logit, -20.0))
