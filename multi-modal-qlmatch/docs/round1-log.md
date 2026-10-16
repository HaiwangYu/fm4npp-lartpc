# Round-1 working log (decisions, findings, deviations from the plan)

## 2026-10-15

### Data and infrastructure
- **Downloads.** Inclusive NoWire (18 files, 194.9 GB) and νe NoWire (18 of 20 files) are md5-verified. Data files are read-only.
  - Zenodo throttles each connection, so the downloads used parallel HTTP byte ranges.
  - A stale downloader left on `polaris-login-02` after a disconnect corrupted some partial ranges by appending to them alongside the new downloader. Those ranges were discarded and re-fetched. The final md5 checks pass.
- **Conversion** (`qlfm.data.convert`). The 753,467 inclusive events go into 108 parts, 146 GB of column-wise `.npy` files. This takes about 3 min on one compute node; on the login node it is roughly 20× slower.
- **DDP hang.** Under `mpiexec` the long `TMPDIR` path broke DataLoader worker sockets (`AF_UNIX path too long`). Fixed by setting `TMPDIR=/tmp` in `scripts/env.sh`.
- **Queue limits.** `debug` allows 1 running and 1 queued job per user (up to 2 nodes). `debug-scaling` allows 1 queued job (up to 10 nodes).
  - Several 2-node pretraining runs are therefore packed into one multi-node job (`scripts/pbs/pretrain_multi.pbs`).
  - Many single-GPU downstream runs share nodes through `scripts/run_queue.sh`.

### Audit (inclusive sample; `fm4npp-lartpc-work/audit_bnb/audit.json`)
- **Occupancy.**
  - Hits per event: median 15.4k (1st–99th percentile 6.6k–28.5k).
  - Charge tokens per event (16 wires × 64 ticks cells): median 2.0k, 99th percentile 3.4k. Only 0.2% of events exceed 4096.
- **Neutrino location.** 44% of events have the ν vertex in the active volume (AV) and 42% in the fiducial volume (FV). In 21% of events the neutrino leaves ≤ 5 hits.
- **Timing.** Beam-flash time minus ν time = 0.478 µs, with a 5–95% range of 0.457–0.507 µs. Some ophits fall at t < 0 or t ≫ 23.4 µs; these are dropped.
- **Task pools.**
  - T2: 294k positive, 158k negative, 302k ambiguous (excluded).
  - T3: 222.7k νμCC, 1,484 νeCC (0.5%), 54.8k NC.
  - T4: 224k events.
- **Baselines.**
  - T2 AUC: Pandora max `nu_score` 0.818; beam-flash PE in 3–6 µs 0.911.
  - T1 (charge-weighted): Pandora ν-slice hits give efficiency 0.77, purity 0.53, F1 0.63.

### MAE charge-target problem (important finding)
- **Symptom.** The first design gave tokens their cell-centre position and used a 4×8 in-cell charge sub-grid plus (log n_hits, log Σq) as the reconstruction target. The masked-charge loss never beat a per-dimension mean predictor, while the light reconstruction learned well.
- **Diagnosis.**
  1. Single-batch overfitting and a "target-leak" plumbing test showed the wiring was correct.
  2. Fourier, learned-grid and combined position embeddings made no difference.
  3. A non-neural baseline was the key result. Predicting a masked cell's statistics from the average of its visible neighbours is *worse* than the global mean, at 8×32, 16×64 and 32×128 cell sizes. Tracks cross a fixed grid at arbitrary offsets: a cell whose corner a track clips gets 1–2 hits, while one it crosses through the middle gets about 16. Given only the grid position, the content of a masked cell is essentially unpredictable.
- **Fix.** Use the Point-MAE / PoLAr-MAE convention: the token position is the patch centre on the track, here the charge-weighted centroid. Encode it with Fourier features (10 octaves) plus an explicit embedding of the centroid's offset inside its cell. Standardize every target dimension within the batch.
- **Result.** On held-out events the masked-charge error falls to 0.61 of the mean baseline (sub-grid 0.61, count/charge 0.52); before the fix it was 0.97–0.99.
- **Defaults now.** `--pos-mode centroid --pos-freq 10 --target-norm 1`.

### E1 scratch sanity (1% labels; ~2–4k labeled events)

| Task | Metric | Result |
|---|---|---|
| T2, charge + light (CL) | AUC | 0.958 |
| T2, charge only (C) | AUC | 0.605 |
| T1, CL | charge-weighted F1 | 0.81 (Pandora 0.63) |
| T4, CL | energy resolution | 37% |

- The large gap between C and CL on T2 is physical. Charge alone has to find a small neutrino among about 26 cosmic muons, whereas the in-time flash flags a neutrino event directly.

### Review decisions (issue #1) and the changes they triggered
- **(a) νe sample merged into every split.** The corpus is now `cache/bnb,cache/nue`: 830,807 events, including 77,340 from the νe sample. Splits are `cache/splits_all.npz`, hashed on (run, subrun) as before.
  - νe-sample share by split: train 9.4%, val 9.2%, test 7.2%.
  - The pretraining scaling subsets and label pools draw from the merged train split, so they contain about 9% νe-sample events.
  - T3 train is now νμCC 200k / νeCC 68k / NC 49k.
- **(b) T2 is now 3-class.**

  | Class | Definition | Train events |
  |---|---|---|
  | 0 | no ν charge (≤ 5 ν hits) | 142k |
  | 1 | visible in-AV ν (≥ 30 collection-plane ν hits) | 333k |
  | 2 | out-of-AV ν that leaves > 5 hits | 241k |

  - The headline score is unchanged: the AUC of in-AV ν against no-ν events, from p(class 1).
  - Also reported: the in-AV vs out-of-AV AUC, 3-class macro-F1 and the confusion matrix.
  - In-AV events with low visibility (6–29 collection-plane hits; 34k) stay excluded.
- **(c, d)** Fixed pretraining steps across data fractions, and the current label thresholds, are both confirmed.
- **Superseded results.** Results computed on the bnb-only corpus moved to `fm4npp-lartpc-work/archive_bnbonly/`. Pretraining had not started yet and was resubmitted on the merged corpus.

### Main pretraining (1 node per run; 17k steps at global batch 256, about 4.4M events)
- FM-C@90%: validation masked-charge loss 0.366, in units of the mean-predictor baseline (1.0).
- FM-CL@90%:
  - masked charge 0.440
  - masked light 0.446
  - light predicted from charge alone ("charge→light") 0.556, against a mean baseline of about 1.13
- Both runs took about 45 min on one node.

### Downstream protocol fixes (v1 results archived to `archive_v1_e3_biasedval/`)
- **Biased val/test subsets.** v1 took the first N eligible events in index order. The νe sample sits at the end of the merged index, so the val subsets (3k events) contained almost no νe-sample events: the flavor val set had 16 νeCC. The event-level and hit test subsets were likewise dominated by inclusive events. Subsets are now drawn at random with fixed seeds.
- **Undertrained scratch and fine-tune runs.** The 6000-step cap left large-budget scratch runs undertrained; charge-only event-level heads had barely started learning. The cap is raised to 15k steps. The queue starts these long runs only in a slot with at least 40 min left, and they sit at the top of the task list.
- **Early v1 signal** at 0.1% labels, before the fixes:

  | Task | CL fine-tune | CL frozen | CL scratch |
  |---|---|---|---|
  | T2 AUC | 0.974 | 0.951 | 0.939 |
  | flavor macro-F1 | 0.512 | 0.417 | 0.340 |
  | T4 resolution | 0.417 | 0.470 | 0.465 |
  | T1 F1 | 0.754 | 0.489 | 0.19 |

### 2026-10-16: E3 complete, E4 complete, E5 checks
- **E3** (192 runs) is in `docs/figures/e3_summary.md` and `e3_label_efficiency.png`.
- **Flavor shortcut check.** Most νeCC training events come from the νe sample, so the model might identify the sample rather than the flavor. It does not appear to: on test events, νeCC efficiency for inclusive-sample νe (n=79) is equal to or higher than for νe-sample events (n=2849):

  | Model | Inclusive-sample νe | νe-sample |
  |---|---|---|
  | FT-CL | 0.86 | 0.80 |
  | frozen-CL | 0.72 | 0.68 |
  | FT-C | 0.70 | 0.64 |
  | scratch-CL | 0.28 | 0.29 |

- **E5 learning-rate scan.** Charge-only (C) scratch at 10% labels, with lr 1e-4 and 1e-3 (default 3e-4):
  - flavor: 0.277 and 0.267
  - event ν/cosmic AUC: 0.692 and 0.561

  Neither beats the default lr (0.277 / 0.769), so the failure of C scratch on event-level tasks is not an lr artifact.
- **E5 layer decay 0.9 for hit fine-tuning, 0.1% labels.** CL: 0.788 (0.773 with the default 0.75). C: 0.001 (unchanged).
- **E4 data scaling (frozen probes).**
  - CL improves from 1% to 10% of the pretraining data, then plateaus: 10% ≈ 50% ≈ 90% within single-seed jitter.
  - C is flat.
  - The pretraining val loss shows the same shape.
  - The likely cause is compute/capacity: every run sees about 4.4M events on a 6.6M-parameter model.
  - Figure: `docs/figures/e4_scaling.png`.
- **E6 (iteration on Q3).** A 21M model (d=384, depth 12) at 2× steps (34k), CL pretrained on 10% and 90%, followed by frozen and fine-tune probes. It tests whether data scaling reappears with more capacity and compute.

### 2026-10-16: E5 and E6 final; round 1 concluded
- **E5, hit fine-tuning with layer decay 0.9 at 10% labels:** C 0.952 (was 0.924 at 0.75), CL 0.960 (was 0.948). This is level with scratch (0.959 / 0.966).
- **E6, larger model** (d=384, 12 layers, 21M params, 34k steps), CL on 10% and 90% of the data.
  - Pretraining validation losses:

    | Data | Masked charge | Masked light | Charge→light |
    |---|---|---|---|
    | 10% | 0.328 | 0.431 | 0.539 |
    | 90% | 0.330 | 0.413 | 0.512 |

  - Every probe improves a lot over the small model. Frozen T1 F1 goes from 0.73 to 0.91; frozen T3 macro-F1 at 10% labels from 0.66 to 0.76.
  - Pretraining on 90% beats 10% in all 9 non-saturated comparisons.
- **Tech note:** `docs/technote-round1.md`. Total compute is about 29 node-hours.
