# Round 2 working log: comparison with reference algorithms

Plan: `docs/plan-round2.md`. Survey: `docs/round2-survey.md`. Tracking: GitHub issue #3.

## 2026-10-17

### Decisions (user)
- Budget: ≤ 50 node-hours.
- Equal pretraining node-hours for every reference, about 4 per run.
- Scope: Pandora references (A); Panda v1 (B1) and PoLAr-MAE (B2), both through `pimm`; qlfm-SD, the Panda-v2-inspired variant (B3); FM4NPP (B4).

### Part A: open-data Pandora references (done)
- Per-event Pandora features (`qlfm/data/pandora_features.py`), plus a summary of the light.
- `scripts/pandora_refs.py` produces two references:
  - **Direct Pandora output:** `nu_score`, a `slice_pdg`-based flavor hypothesis, and calorimetric energy.
  - **Pandora + BDT:** a gradient-boosted classifier on those features, at the same label pools, seeds and test subsets as the neural models.
- **Result:** the BDT is a strong, label-efficient reference.

  | Task | 0.1% labels | 1% | 10% | 100% |
  |---|---|---|---|---|
  | T3 macro-F1, BDT (+ light) | 0.54 (0.56) | 0.62 | 0.64 | 0.66 (0.67) |
  | T3 macro-F1, qlfm FT-CL | 0.51 | 0.66 | 0.72 | 0.77 |
  | T4 energy resolution, BDT + light | 0.400 | 0.342 | 0.308 | 0.291 |
  | T2 AUC, BDT + light | 0.957 | 0.976 | 0.983 | 0.983 |

  qlfm FT-CL reaches T2 AUC 0.969–0.999 over the same budgets.

### Environment checks
- **`mamba_ssm` 2.2.5** runs on Polaris through a venv overlay (`tokenizers>=0.22`). Mamba2 at d=256 with 12 layers processes about 0.9M tokens/s per A100 at 16k-token sequences.
- **`pimm`** (uv environment: torch 2.10/cu126, spconv 2.3.8, flash-attn 2.8.3). The PTv3 backward pass works on Polaris. Batch 48 per GPU runs out of memory; batch 16 fits.

### Implementation notes
- **Common backbone interface** (`qlfm/ref`): `--backbone {qlfm, fm4npp, panda, polarmae}` and `--late-light`, so every model gets identical heads and identical late-fused PMT light.
- **FM4NPP port.** Faithful to their `MambaGPT`. Three deviations forced by our data:
  - **Ordering:** a per-hit time-major raster.
  - **Neighbours:** taken on the same plane.
  - **Relative-offset targets** (neighbour − hit, in units of 10 cm). The smoke test showed that absolute normalized targets are almost solved by copying the input in our geometry (loss 0.004 after 75 steps) and sit below bf16 resolution. The output head runs in fp32.
- **`pimm` reader** (`MicroBooNEPlaneDataset`): one plane of one event is one 2D cloud.
- **Panda configuration:** size-matched to about 21M; in-plane augmentations only; mask patches rescaled to the same physical size.
- **PoLAr-MAE configuration:** ViT-S in fp32, as published; ball radius 4 cm, with up to 1,024 groups per plane.
- **qlfm-SD** (`qlfm/models/sd.py`): Panda-style self-distillation on qlfm tokens. Masked tokens are dropped and only observable tokens are distilled. The EMA teacher is saved as the backbone.

### Incidents and fixes
1. **Bundle job 7756529 cancelled about 7 minutes in** (about 0.7 node-hours lost).
   - Both qlfm-SD runs ran out of memory at batch 32 per GPU with 8 views on the 21M model. Fix: batch 8 per GPU.
   - Panda and PoLAr-MAE ran faster than the gate estimate (0.43 and 0.16 s per iteration), so their schedules would have used only about 2.2 h and 1.5 h of the agreed budget. `pimm` restores the schedule length on resume, so it cannot be extended mid-run. Both were restarted with schedule lengths derived from the measured rates: Panda 1.84M plane-clouds; PoLAr-MAE 3 × 1.67M. Each fills about 3.5 h of training across four 1-hour jobs.
2. **Budget-based schedules for our own trainer.** `train_pretrain` gained `--budget-h`: the run measures its step rate after 200 steps, fixes the total number of steps, and stores it in the checkpoint so resumed jobs keep it. qlfm-SD uses this.
3. **Rank detection under `torchrun`.** `mpiexec` left `PMI_*` variables in the environment, which would have made every rank believe it was alone. `dist.init` now prefers `torchrun`'s `RANK`/`WORLD_SIZE`, and the bundle commands unset `PMI_*`.
