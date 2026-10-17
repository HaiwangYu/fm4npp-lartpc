# Round 2 plan: comparison with reference algorithms

*Written 2026-10-17. Survey: `docs/round2-survey.md`. Round 1: `docs/technote-round1.md`. Tracked in a dedicated GitHub issue.*

## 0. Goal and principles
Compare our model (qlfm: the 6.6M and 21M charge+light MAE backbones from round 1) against:
- **(A)** the reconstruction already in the open data (Pandora), and
- **(B)** four published LArTPC/TPC foundation-model methods: PoLAr-MAE, Panda, Panda v2 and FM4NPP.

All comparisons use the same data, splits, tasks and metrics as round 1 (T1–T4, `labels.py`, `splits_all.npz`).

**Fairness rules:**
1. **Same downstream protocol for every backbone.**
   - Same heads: attentive pooling with 4 queries for T2–T4, and the per-hit MLP for T1.
   - Same label budgets, seeds, early stopping and test subsets.
2. **Light handled the same way.** The reference methods are charge-only. For each backbone we report:
   - (a) charge-only;
   - (b) the charge backbone plus **late-fused** PMT tokens, through one shared light encoder inside the head.

   qlfm-C gets the same late fusion, so it serves as the like-for-like control. qlfm-CL (early fusion, light inside the backbone) is reported as our proposed variant.
3. **Matched pretraining compute.**
   - The primary comparison gives every reference roughly the same pretraining budget: **about 4 node-hours (≈ 2× qlfm-21M)**.
   - We report events, tokens and node-hours seen.
   - Where the published schedule is affordable, an extra "longer" run is labelled as such.
4. **Faithful to the paper unless the data forces a change.** Every deviation is listed in the issue and the note: 2D planes instead of 3D, serialization, losses, sizes. Panda v2 cannot be reproduced (no code), so we only test a qlfm variant built from its design hints and do not call it Panda v2.

## 1. Part A: open-data reconstruction references (CPU, about 1 node-hour)

- **A0. Extraction** (`qlfm/data/pandora_features.py`). Re-read the Pandora tables for every event in the merged corpus and build per-event features:
  - best ν slice: `nu_score`, `slice_pdg`, `flashmatch_score`, vertex;
  - n_PFP, n_tracks, n_showers, max/min `trkshr_score`;
  - slice hits and summed charge per plane;
  - slice time extent;
  - optional per-PFP hit counts.

  Light summary features (for the "+light" variant): beam-flash PE, flash time, flash y/z centre and width, and the slice-vs-flash z distance.
- **A1. Direct Pandora outputs** (no training):

  | Task | Pandora reference |
  |---|---|
  | T1 | ν-slice hits (already measured) |
  | T2 | max `nu_score` (already measured), and `nu_score` combined with `flashmatch_score` |
  | T3 | `slice_pdg` mapped as 14 → νμCC, 12 → νeCC, no ν slice → "NC/other". Report 3-class macro-F1 and the νe/νμ confusion among CC events |
  | T4 | calorimetric energy: collection-plane charge summed over the ν slice, with a linear calibration (log space, 2 parameters fitted on train) |

- **A2. "Pandora + BDT"**: the classic reco-plus-ML analysis reference.
  - Model: `HistGradientBoosting` on the A0 features, trained at the same label budgets (0.1/1/10/100%), 3 seeds at ≤1%.
  - Two variants: charge-only features, and + light summary features.
  - Covers T2, T3 and T4. For T1, the hit-level reference stays Pandora's ν-slice hits.

## 2. Part B: foundation-model references

### B0. Common integration layer (`qlfm/ref/`)
- A backbone wrapper interface that outputs:
  - per-hit features, for T1;
  - token features with event indices, for pooling.
- `train_downstream.py` gets `--backbone {qlfm, panda, polarmae, fm4npp}`, plus `--late-light` (PMT tokens → small MLP, concatenated into the head's attention-pool keys).
- Pretraining entry points:
  - `pimm` methods use their own trainers, launched from PBS via `torchrun`, with data supplied through a `pimm` reader over our cache.
  - FM4NPP uses a port of their `MambaGPT` and objective into our framework.

### B1. Panda v1 (main external reference)
- **Code:** `pimm`; Sonata-v1m1 objective with a PTv3 backbone.
- **Input:** per-plane 2D hit clouds (wire·0.3 cm, x_drift, 0) with features [log q, rms, plane one-hot]. One backbone shared by U, V and Y. Grid about 3 mm; in-plane augmentations only; mask patch rescaled.
- **Size:** about 21M (channels about (32, 64, 128, 256, 384)) to match qlfm-21M. The published 91M configuration is run only if budget remains.
- **Gate:**
  - install check (spconv, flash-attn and pointops on Polaris);
  - one debug job measuring throughput and checking for collapse (prototype usage, feature std).
- **Pretraining:** about 4 node-hours (matched), plus an optional longer run.

### B2. PoLAr-MAE
- **Code:** the `pimm` port: ViT-S, C-NMS ball groups, Chamfer + per-point energy loss.
- **Input:** the same per-plane 2D clouds as B1. The energy target is log-normalised hit charge.
- **Context:** about 1,024 groups per plane [to be measured].
- **Pretraining:** about 4 node-hours.

### B3. Panda v2 → "qlfm-SD" (objective control, not a reproduction)
Panda-style prototype self-distillation inside qlfm's own tokenizer and Transformer:
- EMA teacher;
- multi-crop over cell tokens;
- **masked tokens dropped, and only observable tokens distilled** (the Panda v2 hint);
- 4,096 prototypes, Sinkhorn.

Together with B1 this separates "objective" (MAE vs self-distillation) from "point-native backbone and tokenizer".

- **Pretraining:** about 4 node-hours, CL and C.

### B4. FM4NPP
- **Code:** a port of FM4NPP's `MambaGPT` (Mamba2, causal, RMSNorm, NeRF-style position embedding) into `qlfm/ref/fm4npp.py`. It runs on the stock `mamba_ssm` 2.2.5 through a venv with the `tokenizers` fix.
- **Tokens:** one per hit (about 15k per event). Serialization:
  - time-major boxes (about 8 time × 8 wire per plane);
  - planes interleaved within each time slab;
  - hits sorted along wire (beam-z) inside each box.
- **Objective:** k-NNN, k = 30, neighbours **within the same plane**, among hits later in the scan order, targets precomputed with a GPU KD-tree. Event-difficulty reweighting as in the paper.
  - Deviation: we predict relative offsets rather than absolute positions; this is labelled.
- **Size:** m3 configuration (d = 256, 12 layers, about 5.3M). Optionally m4 (21M).
- **Gate:** one debug job benchmarking forward/backward at 2k, 8k and 16k tokens. If throughput is below about 0.3M tokens/s per GPU, cap the hits per event (e.g. a 6k-hit window around the in-time drift region) and document it.
- **Pretraining:** about 4 node-hours matched (report tokens seen against FM4NPP m3's 21G).

### B5. Zero-shot Panda-Base (optional, under 1 node-hour)
The frozen 3D-simulation weights from Hugging Face, applied to the planar clouds. This gives an "off-the-shelf FM" row.

## 3. Evaluation matrix

For each backbone (Panda, PoLAr-MAE, qlfm-SD, FM4NPP, plus qlfm-C, qlfm-CL and qlfm-M as references from round 1):

| Mode | Label budgets | Seeds | Light | Runs per backbone |
|---|---|---|---|---|
| frozen | 0.1, 1, 10% | 3, 3, 1 | late light | 4 tasks × 7 = 28 |
| fine-tune | 0.1, 1% | 3, 3 | late light | 4 × 6 = 24 |
| frozen, charge-only | 1% | 1 | none | 4 |

- qlfm-C additionally gets late-light runs, since round 1 only had C and CL without late fusion. That is about 28 runs.
- **Total:** about 56 runs per reference backbone, about 260 runs overall. At the round-1 average of about 8 GPU-minutes per run that is about 9–12 node-hours.
- **Metrics:** the same as round 1 (headline metric per task). We add a single comparison table and label-efficiency plots per task with every backbone overlaid.

## 4. Schedule and compute

| Step | Content | Est. node-hours |
|---|---|---|
| A0–A2 | Pandora extraction, direct references, BDT (CPU on a debug node) | 1 |
| G | Environment and throughput gates: `pimm` (spconv, flash-attn, pointops) and the `mamba_ssm` venv; 1 debug job each | 2–3 |
| B0 | Integration layer (code), smoke tests | 1 |
| B1 | Panda pretraining, matched | 4 (+ optional longer run 8–15) |
| B2 | PoLAr-MAE pretraining, matched | 4 |
| B3 | qlfm-SD pretraining (CL and C) | 4–8 |
| B4 | FM4NPP pretraining, matched | 4 (+ gate-dependent) |
| B5 | Zero-shot Panda-Base probes | ≤1 |
| E | Downstream matrix (§3) | 10–14 |
| R | Reserve for reruns and fixes | 8 |
| **Total** | | **≈ 40–50 (≈ 60–70 with the optional longer runs)** |

- **Ordering:** A (fast, informative), then the gates G, then B0, then B1/B4/B3/B2 pretraining in parallel across the two queues, then E.
- **Constraints:** the allocation (suballocation `debug`, about 183 node-hours left) **expires 2026-11-01**. The plan fits in about 1–1.5 weeks of wall time with 1-node jobs and the feeder.

## 5. Deliverables
- Code: `qlfm/data/pandora_features.py`, `qlfm/ref/*`, `--backbone` and `--late-light` options, and `pimm`/FM4NPP launch scripts.
- Results: a unified comparison table and figures in `docs/figures/round2_*`.
- Tech note: `docs/technote-round2.md` and a PDF, with the deviations table for every adapted method.
- The GitHub issue updated at each milestone with job configurations and purposes.

## 6. Risks and fallbacks
- **`pimm` dependencies fail on Polaris** (spconv/CUDA): use a CPU-built spconv or the `pimm` container if available. Otherwise port the Sonata objective into qlfm (B3 already covers this) and use PTv3 without sparse-conv positional encoding.
- **Mamba2 too slow for 15k tokens:** cap the window; or use a Transformer stand-in with the same objective, labelled as such.
- **Self-distillation collapse in the cosmic-dominated domain:** monitor prototype usage and feature std, adjust temperatures or crops once, then report.
- **Time:** if the allocation deadline gets close, drop B5 and the optional longer runs first, then B2.
