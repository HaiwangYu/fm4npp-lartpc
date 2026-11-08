# Paper campaign: multi-modal (charge + light) FM for LArTPC

Tracking doc for the paper campaign (started 2026-11-06). GitHub issue: https://github.com/HaiwangYu/fm4npp-lartpc/issues/7

- Paper plan / initial draft: `docs/paper-plan.pdf` (source `docs/paper-plan-tex/`). Figures and numbers marked PLACEHOLDER come from tech notes 1–4.
- Background: `docs/survey-round1.pdf` (multi-modal FM positioning, experiments), Josh's `ssm-survey-round1.pdf` (SSM backbones; his fork).
- Account: `neutrinoGPU::WireCellDev` (2,000 nh), preemptable queue. Campaign cap for paper Sec. 4: **200 nh**.

## Paper structure (paper section ↔ item in the original plan)

| Paper section | Plan item | Status |
|---|---|---|
| 3. Light helps; learning the correlation is key | 1 | numbers from TN1–4 as placeholders; updated metrics + multi-task FT pending |
| 4. Which MMFM architecture suits LArTPCs? | 2 | **first**: job plan below |
| 5. How the correlations are learned | 3 | after Sec. 4 |
| 6. Scaling | 4 | **on hold** until Sec. 3–5 are done and the go-ahead is given |
| Appendix: model setups, displays | 5 | with the final runs |

## Decisions

| Date | Decision |
|---|---|
| 2026-11-06 | Do paper Sec. 4 (architecture comparison) first, ≤200 nh. |
| 2026-11-06 | All four recipes pretrained at 20 nh each, including a new qlfm-M CL run (old one ≈2 nh; round-2 refs ≈4 nh). |
| 2026-11-06 | T4 scored on a truth-based subset common to all models: CC, fiducial, visible ν, **ν share of event charge ≥ 5%** (keeps 69%; median share is 7%) and **visible fraction (ν deposited energy / E_ν) ≥ 0.3** (containment proxy). All-eligible T4 in the appendix. |
| 2026-11-06 | JEPA is an objective on the qlfm Transformer (MAE vs JEPA isolates the objective). |
| 2026-11-06 | "Reco-3D" ablation becomes an oracle study with truth deposits (open data has no space points). |
| 2026-11-06 | Name the reference "PTv3 + Sonata (as used in Panda)" in the paper. |
| 2026-11-06 | Release the benchmark (splits, tasks T1–T5, metrics, baselines incl. Pandora + BDT). |
| 2026-11-06 | Josh's code audit (`audit-round1.md`, his fork) is fixed before any campaign run; his A1–A5 checks (survey-round1.5) go into paper Sec. 5. |
| 2026-11-06 | Step-0 order: audit fixes → multi-task FT (tested on the existing qlfm-M CL / C checkpoints, also paper Sec. 3) → qlfm-M CL 20 nh → JEPA, PTv3 + L, Mamba-2 + L. |
| 2026-11-06 | Venue: ICML 2027 main conference, Application-Driven ML area (deadline expected late Jan 2027). Fallbacks: ICML 2027 workshop, NeurIPS 2027 Datasets & Benchmarks, MLST. |

## Updated metrics (paper Sec. 3)

Applied to saved test predictions (`preds_test.npz` keeps `gidx`), so training is unchanged.

- T1: per-event ν-hit F1, **unweighted** median over events.
- T2: AUC in-AV ν vs no-ν on events passing the beam-flash cut (TN4: 200 PE; re-derived on val, audit #4).
- T3: macro-F1 (as TN1).
- T4: resolution on the truth-based clean subset: ν share ≥ 5%, visible fraction ≥ 0.3 (above).
- T5: test MSE (log1p PE), total-PE resolution on ≥200 PE events.

## Paper Sec. 4 job plan (≤200 nh)

Recipes (≈20M parameters, light in pretraining):

| Recipe | Backbone | Objective | Light in pretraining | New code |
|---|---|---|---|---|
| qlfm-M CL | Transformer | MAE | PMT tokens (as now) | none |
| qlfm-M CL JEPA | Transformer | I-JEPA-style | same tokens; predict EMA-target embeddings of masked charge blocks + PMTs | objective |
| PTv3 + Sonata + L | PTv3 | self-distillation | event-level cloud: 3 planes + 32 PMT points | data/config |
| Mamba-2 + L | Mamba-2 (FM4NPP m4) | k-NNN + masked light | PMT prefix + suffix tokens | data/loss |

Selection metric: multi-task fine-tune at 1% labels (3 seeds), all five tasks; also frozen at 0.1% and 1%. Convergence: validation loss vs epoch (own objective) and a frozen probe (T2 bright AUC, T5 MSE) at 2/5/10/20 nh checkpoints.

Budget (nh): code + smoke 13 · pretraining 4 × 20 = 80 · probe curve 6 · frozen heads 6 · multi-task FT 16 · single-task FT 2 · reruns/preemption 30 → **≈155** (45 spare for C-only twins or the fusion-depth ablation).

### Checklist

- [x] Step 0: audit fixes — #1 FM4NPP adapter trains when frozen; #2 `final_step`/`truncated` in metrics.json, `--resume 1` default; #3 random pretraining val subset; #5 token cap from checkpoint; #6 PTv3 inputs matched to pretraining views (`--pimm-prep`); #8 per-key loss averaging; #9 `best.pt` always written; #11 FM4NPP final-norm lr group; budget-mode warmup fixed (`--warmup-steps`). Still open: #4 light cut on val (with the metric scripts), #10 label-threshold note, #12 converter asserts
- [x] Step 0: multi-task fine-tune (`qlfm/train_multitask.py`, all backbones; final checkpoint, per-task masked losses, running loss scale) — smoke-tested on qlfm-M CL/C, Panda, JEPA, Mamba-2 + L
- [ ] Step 0: JEPA objective on qlfm backbone (`qlfm/models/jepa.py`, `--method jepa`) — plain JEPA collapses (tgt_std 0.51→0.29, loss_c 0.004). Job 7820819 (1,600 steps, frozen probes, 1% labels, val): target standardisation + cell-centre predictor positions (`tncell`) keeps the spread (tgt_std 0.79, loss_c 0.035); T2 AUC 0.953 vs MAE 0.949, but T1 F1 0.092 vs MAE 0.177 (tn 0.054). Decision pending (see 2026-11-08 note)
- [x] Step 0: PTv3 + Sonata + L (`pimm_configs/panda_l_uboone.py`, `MicroBooNEEventLightDataset`, backbone `panda_l`) — smoke-tested: 0.5 s per 24-event batch
- [x] Step 0: Mamba-2 + L (`qlfm/ref/fm4npp_light.py`, `--method fm4npp_l`, backbone `fm4npp_l`) — smoke-tested: 77 events/s, 19.5M params
- [x] Step 0: denser val logging; snapshots at 10/25/50% of the run (`--keep-frac`)
- [ ] Step 0: T4 truth-based subset definition; T1 unweighted per-event F1; T2 bright-sample AUC in the summary scripts
- [x] Step 1: smoke tests (jobs 7820550, 7820569, 7820581; 7820692 for JEPA variants)
- [ ] Step 2: pretraining qlfm-M CL 20 nh — job 7820585 (preemptable, `pre/s4_m_cl_mae`, budget 19 h, snapshots 10/25/50%)
- [ ] Step 2: pretraining qlfm-M CL JEPA 20 nh — waiting for the JEPA variant test (job 7820819: target standardisation, VICReg, cell-centre predictor positions)
- [ ] Step 2: pretraining PTv3 + Sonata + L 20 nh — job 7820823 (`pre/s4_ptv3_l`, 3.28M events, snapshots every 5%)
- [ ] Step 2: pretraining Mamba-2 + L 20 nh — job 7820822 (`pre/s4_mamba_l`, budget 19 h, snapshots 10/25/50%)
- [ ] Step 3: probe curves
- [ ] Step 3: frozen heads 0.1% / 1%
- [ ] Step 3: multi-task FT 0.1% / 1% × 3 seeds; single-task FT qlfm-M CL 1%
- [ ] Sec. 4 results into `paper-plan.tex`
- [x] Sec. 3: multi-task FT on the existing qlfm-M CL / C (+XF) / Panda (+XF) checkpoints — job 7820653 (`s3/mt`), 12 runs done (results below)

## Timeline (target)

| Dates | Work |
|---|---|
| week of 9 Nov | Step 0–1; qlfm-M CL 20 nh pretraining starts |
| 16–29 Nov | JEPA, PTv3 + L, Mamba-2 + L pretraining |
| 30 Nov – 13 Dec | downstream, probe curves; Sec. 4 results |
| 14 Dec – 10 Jan | Sec. 3 updates (metrics, multi-task FT), Sec. 5 |
| 11–27 Jan | writing; ICML deadline |

## Sec. 3 multi-task results (job 7820653, test, final checkpoint; T2 on all events, old metric definitions)

| backbone | mode | labels | T1 F1 ↑ | T2 AUC ↑ | T3 F1 ↑ | T4 res ↓ | T5 MSE ↓ |
|---|---|---|---|---|---|---|---|
| qlfm-M CL | finetune | 1% (3 seeds) | 0.918 ± 0.001 | 0.993 | 0.748 | 0.269 | 0.529 |
| qlfm-M C + XF | finetune | 1% (3 seeds) | 0.859 ± 0.002 | 0.990 | 0.685 | 0.322 | 0.584 |
| PTv3 + Sonata + XF | finetune | 1% | 0.839 | 0.988 | 0.789 | 0.261 | 0.592 |
| qlfm-M CL | finetune | 0.1% | 0.879 | 0.985 | 0.658 | 0.338 | 0.642 |
| qlfm-M C + XF | finetune | 0.1% | 0.633 | 0.915 | 0.490 | 0.364 | 0.792 |
| qlfm-M CL | frozen | 1% | 0.894 | 0.997 | 0.687 | 0.307 | 0.521 |
| qlfm-M C + XF | frozen | 1% | 0.721 | 0.984 | 0.642 | 0.350 | 0.657 |
| PTv3 + Sonata + XF | frozen | 1% | 0.740 | 0.990 | 0.756 | 0.300 | 0.635 |

The Mamba-2 + L multi-task fine-tune smoke test ran out of GPU memory (bs 32, 40 GB) and `run_queue.sh` retried it until the job ended (≈7 nh lost). Fixed: tasks that exit with an error are logged to `DIR.fails` and skipped after `MAX_FAIL=2` errors. Mamba-2 + L fine-tuning needs a smaller batch with gradient accumulation.

## Node-hour ledger

| Date | Item | nh |
|---|---|---|
| 2026-11-06 | preemptable smoke test of the new scripts (job 7820275) | 0.08 |
| 2026-11-06 | code smoke tests (7820550, 7820569, 7820581, 7820692) | 1.8 |
| 2026-11-07 | JEPA variant test + probes (7820819, 3 nodes) | 1.6 |
| 2026-11-07 | Sec. 3 multi-task FT (7820653; ≈2 nh of work, ≈7 nh lost to the OOM retry loop) | 9.2 |
| | **total so far** | **12.7** |
