"""Panda (Sonata + PTv3) and PoLAr-MAE backbones from pimm behind the common interface (run in the pimm venv).

Per event, the three wire planes are encoded as separate 2D clouds exactly as in pretraining
(qlfm/ref/pimm_ext): points (wire cm, drift cm, plane * 10 cm), energy = hit integral, NormalizeCoord with the
same centre/scale, log-energy map identical to pimm's LogTransform. Pre-processing runs on the GPU.
  panda    : teacher backbone, up-cast 2 levels and upsample to the grid points (Sonata's return_point path);
             a hit takes the feature of its grid voxel (grid 0.3 cm). Tokens for pooling = all hits.
  polarmae : groups -> masked mini-PointNet -> + learned position -> ViT encoder (no masking);
             tokens for pooling = groups; a hit takes the feature of the nearest group centre.
"""
import glob
import math
import os
import re
import sys

import torch
import torch.nn as nn

PIMM_ROOT = os.environ.get("PIMM_ROOT", "/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/tools/pimm")
if PIMM_ROOT not in sys.path:
    sys.path.insert(0, PIMM_ROOT)

from qlfm import geometry as G  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pimm_configs")
CENTER = torch.tensor([520.0, 130.0, 10.0])
SCALE, PLANE_DZ, EMIN, EMAX = 600.0, 10.0, 10.0, 5000.0


def log_energy(e):
    return 2 * (torch.log10(e + EMIN) - math.log10(EMIN)) / (math.log10(EMAX + EMIN) - math.log10(EMIN)) - 1


def planar_clouds(b):
    """Flat hits -> normalized per-(event, plane) clouds. Returns coord [Nh,3], energy [Nh], cloud id [Nh]."""
    plane = b["hit_plane"]
    coord = torch.stack([b["hit_wire"].float() * G.WIRE_PITCH_CM, (b["hit_time"] - G.TRIG_TICK) * G.TICK_TO_CM,
                         plane.float() * PLANE_DZ], 1)
    coord = (coord - CENTER.to(coord.device)) / SCALE
    cloud = b["hit_evt"] * 3 + plane
    return coord, b["hit_q"].clamp(min=1.0), cloud


def _load_state(model, ckpt_dir):
    from pimm.utils.path import latest_complete_checkpoint
    path = ckpt_dir
    if os.path.isdir(os.path.join(ckpt_dir, "model")):
        path = str(latest_complete_checkpoint(os.path.join(ckpt_dir, "model")))
    if os.path.isdir(path):
        path = os.path.join(path, "weights.pth")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[pimm_backbone] loaded {path}: missing {len(missing)}, unexpected {len(unexpected)}", flush=True)


class PandaBackbone(nn.Module):
    def __init__(self, cfg, ckpt=None):
        super().__init__()
        import pimm.models  # noqa: F401
        from pimm.models.builder import build_model
        sonata = build_model(cfg.model)
        if ckpt:
            _load_state(sonata, ckpt)
        self.backbone = sonata.teacher.backbone if ckpt else sonata.student.backbone
        self.backbone.requires_grad_(True)       # pimm freezes the EMA teacher; fine-tuning must train it
        self.up_cast_level = cfg.model.up_cast_level
        self.grid = 0.3 / SCALE
        ch = cfg.model.backbone.enc_channels
        self.d = sum(ch[-(self.up_cast_level + 1):])
        self.hit_dim = 0
        depths = cfg.model.backbone.enc_depths
        self._cum = [sum(depths[:e]) for e in range(len(depths))]
        self.n_layers = sum(depths) + 1

    def layer_id(self, name):
        m = re.search(r"enc(\d+)\.block(\d+)\.", name)
        return self._cum[int(m.group(1))] + int(m.group(2)) + 1 if m else 0

    def forward(self, b):
        from pimm.models.utils.structure import Point
        coord, e, cloud = planar_clouds(b)
        # grid sampling (sum energy per voxel), clouds kept separate via the cloud id in the key
        gc = torch.floor((coord - coord.min(0).values) / self.grid).long()
        key = ((cloud * 4096 + gc[:, 0]) * 4096 + gc[:, 1]) * 8 + gc[:, 2].clamp(0, 7)
        ukey, inv = torch.unique(key, return_inverse=True)
        nv = len(ukey)
        esum = torch.zeros(nv, device=coord.device).index_add_(0, inv, e)
        first = torch.full((nv,), len(key), device=coord.device, dtype=torch.long).scatter_reduce(
            0, inv, torch.arange(len(key), device=coord.device), "amin")
        vc = coord[first]
        vcloud = cloud[first]
        counts = torch.bincount(vcloud, minlength=int(cloud.max()) + 1)
        nonempty = counts > 0
        offset = torch.cumsum(counts[nonempty], 0)
        vgc = gc[first]
        data = dict(coord=vc, grid_coord=(vgc - vgc.min(0).values).int(),
                    feat=torch.cat([vc, log_energy(esum)[:, None]], 1), offset=offset, grid_size=self.grid)
        point = self.backbone(Point(data))
        for _ in range(self.up_cast_level):
            parent, pinv = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[pinv]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent, pinv = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = point.feat[pinv]
            point = parent
        z = point.feat[inv]                                  # per hit
        idx = torch.arange(len(z), device=z.device)
        return dict(z=z, evt=b["hit_evt"], h_hit=None, hit2tok=idx)


class PoLArMAEBackbone(nn.Module):
    def __init__(self, cfg, ckpt=None):
        super().__init__()
        import pimm.models  # noqa: F401
        from pimm.models.builder import build_model
        self.m = build_model(cfg.model)
        if ckpt:
            _load_state(self.m, ckpt)
        self.d = self.m.mask_token.numel()
        self.hit_dim = 0
        self.n_layers = len(self.m.encoder.blocks) + 1 if hasattr(self.m.encoder, "blocks") else 13

    def layer_id(self, name):
        m = re.search(r"encoder\.blocks\.(\d+)\.", name)
        return int(m.group(1)) + 1 if m else (0 if not name.startswith("m.encoder") else self.n_layers - 1)

    def forward(self, b):
        from pimm.models.polarmae.data import packed_to_batched
        coord, e, cloud = planar_clouds(b)
        order = torch.argsort(cloud, stable=True)
        cl_sorted = cloud[order]
        counts = torch.bincount(cl_sorted)
        nonempty = torch.nonzero(counts).flatten()
        offset = torch.cumsum(counts[nonempty], 0)
        feat = torch.cat([coord, log_energy(e)[:, None]], 1)[order]
        points, lengths = packed_to_batched(feat, offset)
        mdl = self.m
        g = mdl.grouping(points, lengths)
        groups, centers, emb_mask, pmask = g["groups"], g["centers"], g["embedding_mask"], g["point_mask"]
        with torch.autocast("cuda", enabled=False):
            flat = mdl.embedding(groups[emb_mask].float(), pmask[emb_mask].unsqueeze(1))
        tokens = flat.new_zeros(groups.shape[0], groups.shape[1], flat.shape[-1])
        tokens[emb_mask] = flat
        enc = mdl.encoder(tokens, mdl.pos_embed(centers), emb_mask).last_hidden_state   # [C, G, D]
        # tokens for pooling: valid groups, with their event index
        cid = nonempty[:, None].expand_as(emb_mask)
        z = enc[emb_mask]
        tok_evt = torch.div(cid[emb_mask], 3, rounding_mode="floor")
        # per hit: nearest valid group centre within its own cloud
        tok_index = torch.full(emb_mask.shape, -1, dtype=torch.long, device=z.device)
        tok_index[emb_mask] = torch.arange(len(z), device=z.device)
        start = torch.cat([offset.new_zeros(1), offset[:-1]])
        hit2tok_sorted = torch.empty(len(order), dtype=torch.long, device=z.device)
        for ci in range(len(nonempty)):
            s0, s1 = int(start[ci]), int(offset[ci])
            valid = emb_mask[ci]
            if not valid.any():
                hit2tok_sorted[s0:s1] = -1
                continue
            d = torch.cdist(feat[s0:s1, :3].float(), centers[ci][valid].float())
            hit2tok_sorted[s0:s1] = tok_index[ci][valid][d.argmin(1)]
        hit2tok = torch.empty_like(hit2tok_sorted)
        hit2tok[order] = hit2tok_sorted
        return dict(z=z, evt=tok_evt, h_hit=None, hit2tok=hit2tok)


def build(name, args, ckpt):
    from pimm.utils.config import Config
    import qlfm.ref.pimm_ext  # noqa: F401
    cfg = Config.fromfile(os.path.join(CFG_DIR, f"{'panda' if name == 'panda' else 'polarmae'}_uboone.py"))
    return PandaBackbone(cfg, ckpt) if name == "panda" else PoLArMAEBackbone(cfg, ckpt)
