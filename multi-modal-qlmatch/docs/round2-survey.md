# Round 2: survey of reference algorithms

*2026-10-17. Inputs: the MicroBooNE open-data documentation and HDF5 inspection; the PoLAr-MAE, Panda and Panda v2 papers and code; the FM4NPP paper, code and weights. Items marked [unverified] still need to be checked on Polaris.*

## 1. Reconstruction already in the open data

### 1.1 HDF5 NoWire files (what we use)
The HDF5 contains three Pandora tables.

| Table | Content | Relevant task |
|---|---|---|
| `pandoraPrimary_table` | One row per primary: `slice_id`, `slice_pdg`, `nu_score`, `flashmatch_score`, `vtx`. About 32 rows per event; most are cosmics | T2 (nu_score), T3 (slice_pdg) |
| `pandoraPfp_table` | PFParticles of the ν-candidate slice only: `pfp_pdg` (11 = shower-like, 13 = track-like), `trkshr_score`, `vtx`. About 1 row per event on average | T3, T4 (features) |
| `pandoraHit_table` | `slice_id` and `pfp_id` for every hit | T1 (ν-slice hits), T4 (slice charge) |

`slice_pdg` codes:
- 12 = νe-like ν candidate
- 14 = νμ-like ν candidate
- 13 = cosmic
- 11 also appears (about 1% of primaries), although the documentation lists only 12, 13 and 14.

**How informative it is** (5,000 inclusive events; 1,840 are flavor-eligible):
- 42% of events have a ν-candidate slice.
- Flavor-eligible events contain 2.1 PFPs (0.44 showers) on average.
- `slice_pdg` as a flavor hypothesis:

  | True class | 14 (νμ-like) | 12 (νe-like) | no ν slice |
  |---|---|---|---|
  | νμCC | 0.74 | 0.05 | 0.20 |
  | νeCC (n=15) | 0.20 | 0.73 | 0.07 |
  | NC | 0.44 | 0.18 | 0.38 |

  It is a crude two-way flavor hypothesis with no NC class.
- There is **no reconstructed energy**. An energy reference has to be built from ν-slice hit charge (calorimetric sum plus a calibration).

Reference baselines already measured in round 1 (merged corpus):

| Task | Pandora | Beam-flash PE cut |
|---|---|---|
| T1 hit F1 | 0.63–0.65 (eff 0.78, pur 0.56) | — |
| T2 AUC | 0.82 (max `nu_score`) | 0.91 |

### 1.2 art/ROOT files (not used)
- They contain much richer reconstruction: Pandora tracks and showers with calorimetry, particle ID, CRT tags, Kalman tracks and cosmic tags.
- No Wire-Cell or deep-learning reconstruction products are listed.
- Using them needs the uboone LArSoft stack (v08_05) and streaming the 6.4 TB inclusive sample over xrootd, then matching events to our (run, subrun, event) split. That is out of scope for round 2; it is a possible round-3 item.

## 2. Foundation-model references

| | PoLAr-MAE | Panda (v1) | Panda v2 ("Diplomacy") | FM4NPP |
|---|---|---|---|---|
| arXiv | 2502.02558 | 2512.01324 | 2609.00611 | 2508.14087 (ICLR 2026) |
| Data | PILArNet-M: 3D simulated voxels, 3 mm, (x,y,z,E); about 1M events, about 4.3k points/event | same | three separately pretrained models: PILArNet-M, sPHENIX TPC, water Cherenkov | sPHENIX TPC 3D spacepoints (η,φ,r,E); about 10M events, about 834 points/event, capped at 3,200 |
| Tokens | about 400 C-NMS ball groups (≤32 points each, radius 15 mm), mini-PointNet | points, sparse-conv embedding (PTv3) | points, PTv3/LitePT hybrid, PointRoPE | one token per spacepoint |
| Backbone | ViT-S: 384 wide, 12 layers, about 22M params | PTv3 v3m2, 5-stage encoder, about 91M params | 4-stage, about 53M params | Mamba2, causal, no attention; m1–m6 = 0.34M–188M (m3 = 5.3M, m4 = 21M) |
| Objective | MAE, 60% masking; Chamfer loss on masked points plus per-point energy L2 | Sonata-style self-distillation: EMA teacher, 2 global + 6 local crops, masked student, 4,096 prototypes, Sinkhorn | Panda-style, but masked points are *removed* and only observable points are distilled | k-next-nearest-neighbour coordinate regression (k = 30, neighbours at larger r) on a hierarchical raster serialization |
| Pretraining compute | 4×A100, 500k steps | 4×A100-40GB, 208k steps × 48 | not reported | m3: 4 GPUs × 20 h; m4: 8 × 32 h |
| Code / weights | yes, MIT; GitHub and Hugging Face; also in `pimm` | yes; `pimm` (MIT) for training, Hugging Face `Panda-Base` (363 MB) | **no**: README says "code coming soon"; no weights | yes: `FM4NPP/PP_collision` (use the `downstream-reproducibility` branch); weights on Hugging Face (LGPL). An independent audit reproduces tracking ARI 0.86 vs the paper's 0.945 [unverified] |
| Light | no | no | water-Cherenkov PMTs as points (not LArTPC light) | no |
| Results nearest to our tasks | none (track/shower/Michel/delta segmentation) | "interaction" clustering (≈ T1/T2); per-particle PID | none for our tasks; the only energy-like regression it reports (water Cherenkov) loses to the classical fitter | per-point noise tagging (≈ T1); per-point PID |

**Things that hold for all four:**
- None has seen 2D wire hits, real cosmic data or LArTPC light.
- None has event-level flavor or energy tasks.
- Every result on our data is therefore new. Published numbers do not transfer.

### Adaptation options and feasibility
1. **Panda v1 through `pimm` on per-plane 2D hit clouds.**
   - Each hit becomes a point (wire·0.3 cm, x_drift cm, 0) with features [log q, rms, plane]. The backbone is shared across the three planes. `GridSample` with `return_inverse` maps predictions back to every hit, which gives T1.
   - `pimm` already has a reader that turns 2D wire-plane data into points with a plane ID.
   - Changes needed: in-plane rotation only, mask patch scaled to wire-time units, about 3 mm grid.
   - About 5k hits per plane, close to PILArNet's about 4.3k points per event.
   - Effort: 2–4 days.
   - Risks: prototype collapse in a cosmic-dominated domain; spconv and flash-attn wheels on the Polaris driver [unverified]; `pimm` launches via Slurm, so it needs a PBS wrapper.
2. **PoLAr-MAE through `pimm` on per-plane 2D hit clouds.**
   - Same reader as option 1, with C-NMS ball groups (about 600–1,000 per plane [unverified]) and the Chamfer + energy objective.
   - Architecturally it is close to qlfm. The main differences are data-adaptive grouping and a point-level target.
3. **Panda v2: not reproducible.**
   - No code, and the prototype count, temperatures and loss form are unpublished.
   - Usable design hints: drop masked tokens instead of mask tokens; distil only observable points; PMTs as points.
4. **FM4NPP, per-hit Mamba2.**
   - `mamba_ssm` 2.2.5 and `causal_conv1d` 1.5.2 are already in the Polaris `conda/2025-09-25` env. The import fails only because of a `transformers`/`tokenizers` version clash; a `--system-site-packages` venv with `tokenizers>=0.22,<0.23` fixes it. GPU kernels are untested.
   - About 15k hits per event is 5–18× FM4NPP's sequence lengths.
   - The paper's k-NNN targets use an O(N²) neighbour search in the dataloader, so we need offline or GPU KD-tree targets.
   - Proposed serialization: time-major boxes with the three planes interleaved.
   - Throughput must be benchmarked before committing. At the paper's efficiency it would cost about 80 node-hours for 4M events, which is out of budget. At about 0.3–0.8M tokens/s per GPU it would cost about 6–20 node-hours.
5. **Zero-shot frozen Panda-Base** (3D simulation weights) on planar points. A cheap "off-the-shelf FM" row; expect poor transfer.
6. **True 3D from wire crossings** (needed for an unmodified Panda or PoLAr-MAE). Requires tomography; there are severe ghost points with about 26 cosmics per event. Weeks of work: out of scope.
