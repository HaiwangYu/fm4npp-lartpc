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
| 2026-11-06 | T4 scored on a truth-based subset common to all models (CC, fiducial, ν share of event charge ≥ X%, lepton contained; X TBD). |
| 2026-11-06 | JEPA is an objective on the qlfm Transformer (MAE vs JEPA isolates the objective). |
| 2026-11-06 | "Reco-3D" ablation becomes an oracle study with truth deposits (open data has no space points). |
| 2026-11-06 | Name the reference "PTv3 + Sonata (as used in Panda)" in the paper. |
| 2026-11-06 | Venue: ICML 2027 main conference, Application-Driven ML area (deadline expected late Jan 2027). Fallbacks: ICML 2027 workshop, NeurIPS 2027 Datasets & Benchmarks, MLST. |

## Updated metrics (paper Sec. 3)

- T1: per-event ν-hit F1, **unweighted** median over events.
- T2: AUC in-AV ν vs no-ν on events passing the 200 PE beam-flash cut (TN4).
- T3: macro-F1 (as TN1).
- T4: resolution on the truth-based clean subset (above).
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

- [ ] Step 0: multi-task fine-tune mode in `train_downstream` (all backbones)
- [ ] Step 0: JEPA objective on qlfm backbone (+ embedding-rank monitor)
- [ ] Step 0: PTv3 + Sonata + L (event-level cloud with PMT points)
- [ ] Step 0: Mamba-2 + L (PMT prefix/suffix, masked-light loss)
- [ ] Step 0: denser val logging; keep 2/5/10/20 nh checkpoints
- [ ] Step 0: T4 truth-based subset definition; T1 unweighted per-event F1; T2 bright-sample AUC in the summary scripts
- [ ] Step 1: smoke tests (30 min each; throughput → steps for 20 nh; resume after kill)
- [ ] Step 2: pretraining qlfm-M CL 20 nh
- [ ] Step 2: pretraining qlfm-M CL JEPA 20 nh
- [ ] Step 2: pretraining PTv3 + Sonata + L 20 nh
- [ ] Step 2: pretraining Mamba-2 + L 20 nh
- [ ] Step 3: probe curves
- [ ] Step 3: frozen heads 0.1% / 1%
- [ ] Step 3: multi-task FT 0.1% / 1% × 3 seeds; single-task FT qlfm-M CL 1%
- [ ] Sec. 4 results into `paper-plan.tex`

## Timeline (target)

| Dates | Work |
|---|---|
| week of 9 Nov | Step 0–1; qlfm-M CL 20 nh pretraining starts |
| 16–29 Nov | JEPA, PTv3 + L, Mamba-2 + L pretraining |
| 30 Nov – 13 Dec | downstream, probe curves; Sec. 4 results |
| 14 Dec – 10 Jan | Sec. 3 updates (metrics, multi-task FT), Sec. 5 |
| 11–27 Jan | writing; ICML deadline |

## Node-hour ledger

| Date | Item | nh |
|---|---|---|
| 2026-11-06 | preemptable smoke test of the new scripts (job 7820275) | 0.08 |
