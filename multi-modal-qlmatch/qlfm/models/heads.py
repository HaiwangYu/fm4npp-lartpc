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
    """Per-hit logit from the hit embedding (if the backbone provides one), its token's contextual
    feature, and (optionally) a per-event extra vector such as pooled late-fused light."""

    def __init__(self, d, hit_dim=128, hidden=128, extra_dim=0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = mlp([hit_dim + d + extra_dim, hidden, hidden, 1])

    def forward(self, h_hit, z_tok, hit2tok, extra=None):
        m = hit2tok >= 0
        parts = [self.norm(z_tok.float())[hit2tok.clamp(min=0)]]
        if h_hit is not None:
            parts.insert(0, h_hit.float())
        if extra is not None:
            parts.append(extra.float())
        logit = self.net(torch.cat(parts, 1)).squeeze(1)
        return torch.where(m, logit, torch.full_like(logit, -20.0))


class LateLight(nn.Module):
    """Late-fused PMT light, identical for every backbone: 32 per-PMT tokens (time-profile MLP + PMT
    position/ID embedding) trained with the head. Returns tokens with event index and a per-event mean."""

    def __init__(self, d, tok_cfg=None):
        super().__init__()
        from qlfm.models.encoder import LightEmbed
        from qlfm.tokenize import TokCfg
        self.cfg = tok_cfg or TokCfg()
        self.embed = LightEmbed(d, len(self.cfg.light_edges))
        self.modality = nn.Parameter(torch.zeros(d))

    def forward(self, b):
        from qlfm.tokenize import tokenize_light
        lt = tokenize_light(b["oph_ch"], b["oph_t"], b["oph_pe"], b["oph_evt"], b["n_events"], self.cfg)
        z = self.embed(lt) + self.modality
        pooled = scatter_mean(z.float(), lt.evt, dim=0, dim_size=b["n_events"])
        return z, lt.evt, pooled
