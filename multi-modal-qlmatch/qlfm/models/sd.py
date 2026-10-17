"""qlfm-SD: Panda-style prototype self-distillation on qlfm tokens (round-2 B3, "Panda-v2-inspired").

Panda (arXiv:2512.01324) recipe, adapted to qlfm's cell + PMT tokens:
  * EMA teacher (momentum 0.994 -> 1), teacher sees the full event (all tokens)
  * student: 2 global views (drift-time crops of 40-100 % of the event, then masked) + 6 local views (10-40 %)
  * Panda v2 (arXiv:2609.00611) hint: masked tokens are DROPPED (not replaced by mask tokens) and only
    observable tokens are distilled
  * head: MLP -> 256-d L2-normalised embedding -> 4096 weight-normalised prototypes
  * teacher targets: Sinkhorn-Knopp (3 iters), teacher temp 0.04 -> 0.07 (warmup), student temp 0.1
  * loss weights: masked-global 0.5, local 0.5 (Panda: mask 2/8 + roll-mask 2/8, local 4/8)
Token correspondence across views is exact (a cell token is the same token in every view), so no
nearest-neighbour matching is needed.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoHead(nn.Module):
    def __init__(self, d, hidden=2048, emb=256, n_proto=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, emb))
        self.proto = nn.utils.parametrizations.weight_norm(nn.Linear(emb, n_proto, bias=False))
        self.proto.parametrizations.weight.original0.data.fill_(1.0)
        self.proto.parametrizations.weight.original0.requires_grad_(False)   # DINO: fixed prototype norms

    def forward(self, z):
        return self.proto(F.normalize(self.mlp(z.float()), dim=-1))


@torch.no_grad()
def sinkhorn(scores, temp, n_iter=3):
    q = torch.exp((scores - scores.max()) / temp).t()          # [K, N]
    q /= q.sum()
    K, N = q.shape
    for _ in range(n_iter):
        q /= q.sum(1, keepdim=True); q /= K
        q /= q.sum(0, keepdim=True); q /= N
    return (q * N).t()


class SelfDistill(nn.Module):
    def __init__(self, fm, n_global=2, n_local=6, mask_start=0.5, mask_end=0.9, warmup=0.05,
                 t_teacher=(0.04, 0.07), t_student=0.1, momentum=(0.994, 1.0), strip_ticks=256):
        super().__init__()
        self.fm = fm
        self.teacher = copy.deepcopy(fm)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        d = fm.cfg.d
        self.head = ProtoHead(d)
        self.head_t = copy.deepcopy(self.head)
        for p in self.head_t.parameters():
            p.requires_grad_(False)
        self.n_global, self.n_local = n_global, n_local
        self.mask_start, self.mask_end, self.warmup = mask_start, mask_end, warmup
        self.t_teacher, self.t_student, self.momentum = t_teacher, t_student, momentum
        self.strip_ticks = strip_ticks
        self.progress = 0.0                                     # set by the training loop (step / steps)

    # ---------- schedules ----------
    def _ramp(self, a, b):
        return a + (b - a) * min(1.0, self.progress / self.warmup)

    @torch.no_grad()
    def ema_update(self):
        m = self.momentum[0] + (self.momentum[1] - self.momentum[0]) * 0.5 * (1 - math.cos(math.pi * self.progress))
        for ps, pt in zip(list(self.fm.parameters()) + list(self.head.parameters()),
                          list(self.teacher.parameters()) + list(self.head_t.parameters())):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    # ---------- views ----------
    def _crop(self, ct, n, lo, hi):
        """Random drift-time window per event covering a fraction in [lo, hi] of the event's time extent."""
        dev = ct.tok_time.device
        t = ct.tok_time
        tmin = torch.full((n,), 1e9, device=dev).scatter_reduce(0, ct.tok_evt, t, "amin")
        tmax = torch.full((n,), -1e9, device=dev).scatter_reduce(0, ct.tok_evt, t, "amax")
        span = (tmax - tmin).clamp(min=1.0)
        frac = lo + (hi - lo) * torch.rand(n, device=dev)
        start = tmin + (1 - frac) * span * torch.rand(n, device=dev)
        e = ct.tok_evt
        return (t >= start[e]) & (t <= (start + frac * span)[e])

    def _mask(self, ct, n, ratio):
        """Drift-time strips across all planes (half the budget) + random tokens -> True = dropped."""
        dev = ct.tok_time.device
        n_strip = 8192 // self.strip_ticks + 1
        strip = (ct.tok_time / self.strip_ticks).long().clamp(0, n_strip - 1)
        p_strip = ratio * 0.5
        m = torch.rand(n, n_strip, device=dev)[ct.tok_evt, strip] < p_strip
        m |= torch.rand(m.shape, device=dev) < (ratio - p_strip) / (1 - p_strip)
        return m

    def forward(self, b):
        fm, n = self.fm, b["n_events"]
        ct, lt = fm.tokenize(b)
        nc = ct.tok_evt.numel()
        # ---- teacher on the full event ----
        with torch.no_grad():
            xc_t, xl_t, _ = self.teacher.embed(ct, lt)
            (zc_t, zl_t), _ = self.teacher.encode(xc_t, ct.tok_evt, xl_t, lt.evt if lt is not None else None, n)
            zt = torch.cat([zc_t, zl_t]) if zl_t is not None else zc_t
            tt = self._ramp(*self.t_teacher)
            q = sinkhorn(self.head_t(zt), tt)                   # [Ntok, K] soft assignments
        xc, xl, _ = fm.embed(ct, lt)
        ratio = self._ramp(self.mask_start, self.mask_end)
        losses, out = [], {}
        views = [("global", 0.4, 1.0, True)] * self.n_global + [("local", 0.1, 0.4, False)] * self.n_local
        for kind, lo, hi, masked in views:
            keep = self._crop(ct, n, lo, hi)
            if masked:
                keep &= ~self._mask(ct, n, ratio)
            idx = torch.nonzero(keep).flatten()
            use_light = lt is not None and kind == "global"
            (zc, zl), _ = fm.encode(xc[idx], ct.tok_evt[idx], xl if use_light else None,
                                    lt.evt if use_light else None, n)
            zs = torch.cat([zc, zl]) if use_light else zc
            tidx = torch.cat([idx, nc + torch.arange(lt.evt.numel(), device=idx.device)]) if use_light else idx
            logp = F.log_softmax(self.head(zs) / self.t_student, dim=-1)
            l = -(q[tidx] * logp).sum(-1).mean()
            losses.append((0.5 / self.n_global if kind == "global" else 0.5 / self.n_local) * l)
        loss = sum(losses)
        with torch.no_grad():
            usage = q.mean(0)
            out["proto_entropy"] = -(usage * (usage + 1e-12).log()).sum() / math.log(usage.numel())
        out.update(loss=loss, loss_c=loss.detach(), n_tok=torch.tensor(float(nc), device=loss.device))
        return out
