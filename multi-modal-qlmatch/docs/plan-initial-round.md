# Multi-modal (charge + light) LArTPC foundation model — initial-round plan

*Status: plan for round 1 (prototype on MicroBooNE open data). Written 2026-10-15.*
*Companion docs: `microboone_foundation_model_report.html` (feasibility survey, used as reference, not followed verbatim); the tech-note will be written at the end of the round.*

---

## 0. Goal and success criteria

Build a self-supervised (SSL) backbone that ingests **charge (TPC wire hits)** and **light (PMT hits)** from the same readout as one token sequence. Then show that small heads on top of it solve physics tasks better or more label-efficiently than the same network trained from scratch.

The round answers three questions:

| # | Question | Experiment |
|---|---|---|
| Q1 | Does SSL pretraining help? | Same architecture: (a) scratch, (b) frozen FM + head, (c) FM full fine-tune. Swept over labeled-data fractions. |
| Q2 | Does light help? | Charge-only FM (FM-C) vs charge+light FM (FM-CL), with matched architecture, steps and data. Scratch runs also come in C and CL versions. |
| Q3 | How does the FM scale with pretraining data? | FM-CL (and FM-C if budget allows) pretrained on 1%, 10%, 50% and 90% of events, evaluated with a fixed downstream label budget. |

**The round is done when one of these holds:**
- The comparisons are physically sensible and robust, at minimum:
  - frozen or fine-tuned FM beats scratch at small label budgets, and the gap shrinks as labels grow;
  - CL ≥ C, with the largest gain on the event-level ν/cosmic task;
  - downstream quality increases monotonically with pretraining fraction.
- Or further iterations stop moving the metrics: fewer than about 1–2% relative improvement over two consecutive iterations.

The tech-note will then summarize the findings.

**Compute constraint, agreed with the user:**
- ≤ **100 node-hours** total, `debug` / `debug-scaling` queues only, account `neutrinoGPU::debug`.
- Walltime is 1 h, so long runs chain checkpoint/resume jobs.
- Polaris node: 4× A100-40GB, 32-core CPU, 512 GB RAM.

**Paths:**
- Data: `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data`
- Work/outputs: `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work`
- Code: `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch`

---

## 1. Data

### 1.1 Samples (downloading in progress, md5-verified)

| Sample | Zenodo | Files | Events | Use |
|---|---|---|---|---|
| BNB inclusive overlay, NoWire | 8370883 | 18 × ~11 GB (195 GB) | 753,467 | Main corpus: pretraining plus all downstream tasks |
| νe-CC overlay, NoWire | 7261921 | 20 × ~1.7 GB (33 GB) | 89,339 | Optional extra νe-CC labels for the flavor head only (see §1.5) |

Both are **overlays**: GENIE+Geant4 neutrinos mixed with real off-beam cosmic data and noise. Cosmic hits therefore carry no MC truth.

### 1.2 What is used from each event (data contract)
- **Charge, `hit_table`:** `local_plane` (0=U, 1=V, 2=Y), `local_wire`, `local_time` (ticks, 0.5 µs; trigger at tick 800), `integral`, `rms`.
  - About 15.7k hits per event, dominated by cosmics.
- **Light, `ophit_table`:** `hit_channel` (0–31), `peaktime` (µs within the 23.4 µs beam window), `width`, `pe` (clipped at 0).
  - About 360 ophits per event.
- **Light, `opflash`/`opflashsumpe`:** beam flashes and their per-PMT PE.
  - Used **only** for the audit and as the input to a classical baseline.
  - Not given to the model, so a masked-light target can't leak through flash summaries.
- **Truth (labels only, never model input):**
  - `event_table`: `nu_pdg`, `is_cc`, `nu_energy`, `nu_vtx`, `nu_vtx_corr`, `lep_energy`.
  - `edep_table` + `particle_table`: per-hit neutrino association.
- **Baseline only:** Pandora tables (`nu_score`, `flashmatch_score`, slice membership).
- **Indexing:**
  - Each column is an `(N, k)` dataset.
  - `<table>/event_id.seq_cnt` gives `[event_seq, n_rows]` per non-empty event.
  - Rows are contiguous in event order.
  - Joins use `(event, hit_id)` and `(event, g4_id)`.

### 1.3 Audit (step D0, run on one shard before the bulk conversion)
1. **Verify the indexing.** Check that `hit_id == row offset`, the `seq_cnt` semantics, and `parent_id` for primaries (0 or −1).
2. **Distributions.** Histogram hits/event, hits/plane and ophits/event. Measure the occupied-cell (token) count as a function of cell size, which sets the token budget.
3. **Timing.** Compare `opflash.time` with `nu_vtx_wire_time` for active-volume CC events to fix where the neutrino flash sits (expected ≈ 3.6–5.3 µs). Also find the in-time drift window: ticks ≈ 800–5463 are in-time-compatible.
4. **Label statistics.**
   - Fraction of events with ν vertex in active volume and in fiducial volume.
   - Distribution of ν-associated hits and visible ν energy.
   - Class counts for νμCC / νeCC / NC in the fiducial volume.
5. **Event displays.** Reproduce one event display (3 planes + PMT pattern) to confirm geometry. Use the 32-PMT (y, z) map from `ublightmodelnet`/`lardly`, keyed by OpChannel.

### 1.4 Preprocessing (step D1)
Convert HDF5 into a compact, memory-mappable per-shard cache (`.npy` plus offsets):
- Per hit: `plane:u8`, `wire:i16`, `time:i16`, `log_integral:f16`, `rms:f16`, plus labels (`is_nu:u8`, `nu_edep_frac:f16`).
- Per ophit: `ch:u8`, `t:f16`, `width:f16`, `pe:f16`.
- Per event: truth scalars, the Pandora baseline score, and `(run, subrun, event)`.

Estimated size is about 110 GB for the inclusive sample. Conversion runs as a CPU-parallel job, one process per shard.

**Tokenization happens on the GPU at load time**, not in the cache. Cell size and patching can then be changed without re-preprocessing.

### 1.5 Splits and leakage control
- **Split rule:** split by a hash of `(run, subrun)` into **train 90% / val 5% / test 5%**. This keeps reused off-beam cosmic frames from the same subrun inside one partition.
- **Pretraining** uses only the train partition, with **no truth**.
- **Scaling fractions:** the 1%, 10% and 50% subsets are nested random subsets of the train partition. "90%" means the full train partition.
- **Labeled budgets** for downstream tasks are nested subsets of the train partition: **0.1%, 1%, 10%, 100%** of the labeled pool, about 0.7k, 7k, 68k and 680k events. Val and test are never used for pretraining.
- **νe sample:** an optional enrichment for flavor training only. It is restricted to the same active/fiducial-volume selection as the inclusive positives.
  - Test metrics are always reported on the inclusive test split, which has natural class proportions.
  - A "source probe" (predicting which sample an event came from) checks for shortcuts.
  - This is a stretch item; inclusive alone gives O(10³) νe-CC events.

---

## 2. Tokenization: one token space for charge and light

Both modalities become tokens of width *d*, built as

`token = modality_adapter(local measurement set) + position_encoding(native coordinates) + sensor/plane embedding + modality embedding`.

### 2.1 Charge tokens: occupied wire–time cells, one mini-PointNet per cell
- **Cells:** each plane is divided into a grid of **W wires × T ticks**. The starting point is 16 wires × 64 ticks, about 4.8 cm × 1.8 cm of drift.
  - The cell size is tuned in the D0 audit so that the 99th percentile of tokens per event is ≤ 4096.
  - Only occupied cells become tokens (sparse ViT / PoLAr-MAE style, but on a deterministic grid, so no FPS/kNN cost).
- **Per-hit features inside a cell:**
  - relative wire and relative time within the cell;
  - `log(integral)`;
  - `rms`.
- **PointNet:** a shared MLP over the hits, then max- and mean-pooling via `torch_scatter`, giving a token embedding. There is no cap on hits per cell.
  - Per-hit features are kept so the hit-level head can be decoded back to individual hits.
- **Cell-level extra features:** `log(Σ integral)` and the hit count, so energy information survives any normalization.
- **Position:**
  - Fourier features of the cell's charge-weighted centroid, expressed physically:
    - drift coordinate x_app = (t − 800) · 0.0549 cm;
    - wire coordinate in cm along that plane's pitch direction.
  - Plus a learned **plane embedding** (U/V/Y).
  - The time axis is shared across planes, which gives the backbone a common drift coordinate to relate the three views.
- **Overflow:** events above the token budget are randomly subsampled for pretraining, and the overflow rate is logged. Evaluation uses the full event via a larger cap or chunking. The audit decides which.

### 2.2 Light tokens: one token per PMT, carrying a time profile
- **32 tokens per event**, one per PMT (OpChannel).
- **Feature:** the PMT's ophits are histogrammed in **time bins** across the 23.4 µs beam window, as log(1+PE) per bin. This is a coarse "PMT waveform" rebuilt from ophits. Planned binning: 0.5 µs bins around the beam (1–8 µs) and coarser bins elsewhere, about 24 bins in total.
  - Total PE and first-hit time are added as extra features.
- **Position:** Fourier features of the PMT's physical (y, z), with x ≈ −11 cm, plus a learned PMT-ID embedding.
- **Why per-PMT tokens with time profiles:**
  - They keep both observables Wire-Cell QL-matching relies on: the **32-PMT PE pattern** (compared against charge position via a photon library) and **flash timing** (in-time vs out-of-time).
  - They make "mask a PMT" and "mask all light" natural SSL operations.
- **Excluded inputs:** flash-level summaries are deliberately left out of the inputs, since they derive from the same ophits.

### 2.3 Modality and missing-light handling
- A learned modality embedding (charge / light) is added to every token.
- A charge-only model simply has no light tokens.
- The CL model is trained with modality dropout (§4), so it also runs without light.

---

## 3. Backbone

**Choice: a bidirectional Transformer encoder** (pre-LN ViT style) with FlashAttention-2 via `flash_attn_varlen_func`. Packed variable-length sequences mean no padding waste.

**Why a Transformer for round 1, and when Mamba would be worth it:**
- **Token counts suit it.** With 1–4k tokens per event, and only the ~40% visible tokens in the MAE encoder, global attention is cheap on A100.
- **MAE needs order-free context.** MAE-style SSL needs bidirectional, order-free context over an arbitrary visible subset. Mamba needs a serialization (scan order is an inductive bias) and bidirectional scans, and masking breaks the sequence.
- **Mamba's payoff is elsewhere.** Mamba's advantage (FM4NPP) appears with per-hit tokens (~10⁴–10⁵) and very large models. That is a round-2 ablation: per-hit tokens + Mamba2 + wire-time raster serialization, reusing FM4NPP code.
- **Evidence:** third-party point-cloud benchmarks show windowed/flash attention (PTv3) as fast as or faster than Mamba variants at similar quality.
- **Environment:** `mamba_ssm` is not importable in the stock environment, while `flash_attn` 2.8.3 is.

**Prototype size.** The final choice is made after the throughput test.

| Config | Width | Depth | Heads | Encoder params | Role |
|---|---|---|---|---|---|
| S (default) | 256 | 8 | 8 | ~6.5M | All main comparisons |
| M (stretch) | 384 | 12 | 6 | ~21M | Only if S is clearly capacity-limited and budget remains |

- **MAE decoder:** width 192, depth 3. It is discarded after pretraining.
- **Final LayerNorm:** the encoder's final LN is dropped for feature extraction (PoLAr-MAE finding).

---

## 4. Self-supervised objective

### Stage A (primary): multi-modal masked autoencoding (MultiMAE-style)

L = L_charge-MAE + λ_L · L_light-MAE (λ_L chosen by downstream val, start 1.0; average each modality's loss separately first)

**Charge masking.** Mask 60% of charge tokens, as a mix of:
- (i) random cells;
- (ii) **drift-time strips masked jointly across all three planes**. Because time is shared across planes, masking the same drift window everywhere prevents copying a hidden projection from another plane.

The mix starts at 50/50; its effect is ablated only if needed.

**Charge target per masked cell:**
- a fixed sub-grid of the cell, 4 wire × 8 time bins, holding Σ log-charge (a 32-dim "mini-image");
- plus the log hit count and log total charge.

The loss is MSE. As in MAE, the decoder knows the positions of masked cells, so the task is **attribute reconstruction of occupied cells**.

**Light masking.** Per event, one of three modes:
- with probability 0.4, mask ~50% of PMT tokens;
- with probability 0.4, mask **all 32** PMT tokens. This is the charge→light prediction: a learned "photon-library + flash-matching" surrogate;
- with probability 0.2, mask no light.

The light target is each masked PMT's time-profile vector (log1p PE), with MSE loss.

**Charge-only FM (FM-C)** uses the same recipe with no light tokens.

Training settings:
- AdamW, lr 1.5e-4 × (B/256), weight decay 0.05, 5% warmup then cosine decay, bf16.
- Global batch 256–512 events.
- Steps are fixed per experiment (§6). Each run is restartable from checkpoints.

### Stage B (only if frozen probes lag clearly behind fine-tuning)

MAE features are known to be weak under linear and frozen probing. Mitigations, in order of cost:
1. **Attentive probe** instead of a linear one. This is already the default head (§5).
2. **Multi-layer feature concatenation**: the mean of the last 4 blocks.
3. **Masked latent self-distillation** with an EMA teacher, iBOT/JEPA-style (as in Panda). The student predicts the teacher's latent features at masked token positions. This is the main lever if frozen performance is poor.
4. **Pooled charge↔light InfoNCE alignment**, with weight 0.1.

Stage B is tried only if the round-1 criteria aren't met with Stage A.

---

## 5. Downstream tasks and heads

All heads are small (≤ ~0.5M params) and identical across the scratch / frozen / fine-tune modes.

| ID | Task | Label definition (finalised after the D0 audit) | Head | Primary metrics |
|---|---|---|---|---|
| **T1** | Hit-level ν vs cosmic | A hit is ν if its matched edep (max-fraction particle) comes from the ν interaction with non-negligible MC energy; otherwise cosmic/noise. Token loss uses the charge-weighted ν fraction of the cell; per-hit prediction = token logit + per-hit MLP on (hit features ⊕ token embedding). | per-token MLP + per-hit MLP | charge-weighted ν-hit efficiency and purity, F1, AUC; event-level "ν cluster recovered" fraction |
| **T2** | Event-level visible ν vs cosmic-only-like | **Positive:** ν vertex in TPC active volume **and** ≥ N_vis visible ν-associated charge, e.g. ≥ 50 collection-plane ν hits (tuned in the audit). **Negative:** essentially no ν charge (≤ a small threshold), i.e. dirt/cryostat ν. Ambiguous events are dropped from training and reported separately. | attentive pool (1 cross-attn block, 4 queries) → MLP | ROC-AUC, signal efficiency at fixed background rejection, PR-AUC |
| **T3** | Flavor/interaction class | Vertex in fiducial volume (3 cm inside active): **νμCC / νeCC / NC**, 3 classes, ν and anti-ν merged | attentive pool → MLP | macro-F1, per-class efficiency/purity, νe-CC efficiency vs purity, confusion matrix |
| **T4** | Neutrino energy | CC events with vertex in fiducial volume: regress log E_ν (true `nu_energy`) | attentive pool → MLP (Huber on log E) | resolution σ((E_reco−E_true)/E_true), bias, in bins of E_true |

**Modes, for each of the C and CL inputs:**
- **scratch**: the same encoder (+ tokenizer) and head, randomly initialized, trained end-to-end.
- **frozen**: the FM encoder is frozen; only the head is trained. This is the user's "frozen FM + DT fine-tune training".
- **finetune**: FM-initialised, everything trained, with layer-wise lr decay of 0.75.
- **Cheap diagnostic:** a linear probe on frozen mean-pooled features, as a representation-quality check (not a headline result).

**Training protocol:**
- One run per (task group, mode, input, label budget, seed). T3 and T4 share an event-selection definition and can be trained in one multi-head run. T1 and T2 run separately.
- Fixed max steps with early stopping on the val metric. Seeds: 3 for budgets ≤ 1%, 1 above that.
- Several single-GPU runs are packed into one node (4 runs per node) to use the debug queue efficiently.

**Reference baselines (no ML training needed):**
- T1: Pandora — hits in the slice with the highest `nu_score` counted as ν.
- T2: Pandora max `nu_score` per event.
- Classical light cuts: beam-flash PE in the ν time window, and flash z-center vs charge z-centroid (a crude QL-matching proxy, after `Optical Information.ipynb`).
- Wire-Cell's published numbers (arXiv:2101.05076) are quoted for context only, since they use full 4.8 ms readout and cosmic flashes that these files lack.

---

## 6. Experiment matrix and compute budget

| Step | Content | Est. node-hours |
|---|---|---|
| E0 | Env test, throughput and memory profiling, tokenizer and cell-size choice, data-loader speed | 3 |
| D1 | HDF5 → cache conversion (CPU job, 2 nodes in debug, ~1 h) | 2 |
| E1 | Supervised scratch sanity runs: T1/T2 small budget, C and CL. Verifies labels and heads learn. | 4 |
| E2 | Pretrain FM-C@90% and FM-CL@90%, debug-scaling up to 10 nodes chained (or 2 nodes × several h); about 2–4 × 10⁴ steps at batch 512 | 2 × 10 = 20 |
| E3 | Downstream T1–T4 × {scratch, frozen, finetune} × {C, CL} × {0.1, 1, 10, 100%} labels, seeds as in §5, packed 4 per node | 30 |
| E4 | Pretrain FM-CL@{1, 10, 50}% with the same steps as E2 (plus FM-C if budget allows), then frozen probes T1–T4 at a fixed label budget (10%) | 3 × 5 + 5 = 20 |
| Iteration reserve | Fixes, Stage B if needed, reruns | 20 |
| **Total** | | **≈ 100** |

Budget control:
- Every job logs node-hours, and a ledger is kept in `fm4npp-lartpc-work/ledger.md`.
- If the throughput test shows E2 would exceed its share, the default actions are to cut steps or the model to S, and to drop the FM-C scaling arm.

---

## 7. Implementation layout (code repo `multi-modal-qlmatch/`)

```
qlfm/                    python package
  data/
    h5reader.py          seq_cnt-based reader for the open-data HDF5
    convert.py           HDF5 -> per-shard npy cache (hits, ophits, truth, Pandora baseline)
    labels.py            T1-T4 label definitions (single source of truth)
    splits.py            run/subrun hashed splits, nested fraction subsets
    dataset.py           memory-mapped dataset + collate (packed hits/ophits)
  geometry.py            MicroBooNE constants, wire->cm, PMT (y,z) map
  tokenize.py            GPU tokenizer: hits->cells->PointNet tokens; ophits->PMT tokens
  models/
    encoder.py           Transformer (flash-attn varlen), modality/plane/PMT embeddings
    mae.py               masking, decoder, charge/light targets & losses
    heads.py             token MLP, per-hit MLP, attentive pool, linear probe
  train_pretrain.py      DDP pretraining (resume-safe, step-based)
  train_downstream.py    scratch / frozen / finetune; packs runs per GPU
  evaluate.py            metrics, plots, baselines
scripts/                 download, audit, pbs templates, run-matrix launcher
configs/                 yaml per experiment
docs/                    plan, audit notes, tech-note
```

- **Software:** Polaris `conda/2025-09-25` module (PyTorch 2.8, CUDA 12.9, flash-attn 2.8.3, torch_scatter, h5py). No custom environment is needed.
- **Outputs:** everything (checkpoints, logs, metrics JSON, plots) goes to `fm4npp-lartpc-work/<experiment>/<run>`.

---

## 8. Iteration loop and decision points

1. **After E1** (scratch sanity): if the scratch CL T2 AUC isn't clearly above the Pandora-score baseline at the full label budget, revisit the labels and tokenizer before pretraining.
2. **After E2/E3 at the 1% label budget:**
   - If frozen FM < scratch, first check the probe head (attentive vs linear, multi-layer features), then the masking recipe, then Stage B.
   - If CL ≈ C on T2, check the light tokenization (time-bin layout around the beam window, PE normalisation) and λ_L. Then check the charge→light prediction quality directly; it should beat a "mean PMT pattern" baseline.
3. **After E4:** non-monotonic scaling means too few steps at large fractions or overfitting at small ones. Check the pretraining val loss vs fraction before drawing conclusions.
4. **Stop** when the §0 criteria are met or two iterations give < 1–2% relative gains. Then write the tech-note (`docs/technote-round1.md`) covering the data contract, design, results tables and figures, the three comparisons, limitations (overlay-only, no beam-off sample, beam-window-only light) and next steps (Mamba/per-hit ablation, self-distillation, waveform-level data, interaction-level heads).

---

## 9. Known risks and limitations

- **No genuine cosmic-only events.** T2 negatives are overlays whose ν left no visible charge. Some of them still have in-cryostat ν light, which is fine: the model must learn that light without matching charge isn't a TPC ν. Claims are limited to this definition.
- **Beam-window-only light** (23.4 µs). Out-of-time cosmics have no flash, so "QL matching" here is in-window matching only.
- **Hit truth is MC-only.** A hit overlapping a cosmic pulse can carry fraction 1.0 for a tiny MC deposit, so an MC-energy threshold is used and documented.
- **νe sample mixing.** It is optional and guarded by the source probe; headline numbers use the inclusive sample only.
- **Budget.** 100 node-hours forces a small model (S) and limited seeds, so error bars come from seeds at small budgets plus bootstrap over test events.

## 10. Key references
- **Data:** MicroBooNE open samples (Zenodo 8370883, 7261921); arXiv:2309.15362; github.com/uboone/OpenSamples; hdf5maker exporter.
- **Charge–light matching:** Wire-Cell arXiv:2011.01375, arXiv:2101.05076; `wire-cell-toolkit/match`.
- **SSL/FM:**
  - MAE 2111.06377; Point-MAE 2203.06604; PoLAr-MAE 2502.02558; MultiMAE 2204.01678; 4M 2312.06647.
  - FM4NPP 2508.14087; Panda 2512.01324; Panda V2 2609.00611; Sonata 2503.16429; I-JEPA 2301.08243.
  - FASERCal 2604.07037; jets→ν transfer 2604.12364; attentive probing 2506.10178; Perceiver IO 2107.14795.
- **Prior ML on these data:** NuGraph2 arXiv:2403.11872 (hit-level filter/semantic, charge only).
