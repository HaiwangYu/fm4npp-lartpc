# Plan: P3 fine-tuned on real SBND data with Wire-Cell labels (#16, step 3 of the 25 Nov next steps)

Status: **done 2026-11-27** (results in section 8). Related: #14 (real data, E7), #16 (P3), `status-2026-11-25.pdf` (slide 15).

## 1. Why

P3 is the best simulation-trained model on real data, but three gaps remain:
- **Hit level:** P3 + data-like-width fine-tune agrees with Wire-Cell (WC) on 0.81–0.83 of WC's ν charge on held-out beam-on.
- **Off-beam:** it calls 1.9–3.1% of off-beam charge ν (WC 0.44%).
- **Events:** its beam-on excess at WC's off-beam rate is about 35.5 points (WC 39).

E7 showed what a short fine-tune on real data with WC labels does for M CL:
- data agreement 0.70 → 0.75;
- beam-on pass rate 51–52% at WC's off-beam rate;
- simulation T1 lost 0.06 (0.92 → 0.85).

This plan asks:
1. How far does real data close the remaining gap for P3?
2. Does a cleaner label (WC candidates that pass the cosmic tagger) fix the off-beam ν charge?
3. How much simulation performance does it cost?

## 2. What we reuse (E7 recipe, #14)

| item | setting |
|---|---|
| Data | `cache/sbnd_data_m/{bnb-opendata, offbeam-intime-run1}` (sbnd-h5maker 0.6, blobs added). Split by source file 75/5/20 (`cache/splits_sbnd_data.npz`): train 102,979, val 7,925, test 29,012 events. Held-out test files = `cache/sbnd_data_test_keys.npy`. |
| Init | `--init <finished simulation fine-tune>`: all weights incl. heads |
| Tasks | T1 hit = WC footprint; T2 evt_nu = event has a WC hit; T5 light (data's own PMT light) |
| Steps / LR | 3,000 steps, lr 1e-4, layer decay 0.9, `--select last` (E7: about 20 min on one A100) |
| Environment | `@pimm` venv (P3 needs spconv), `QLFM_DETECTOR=sbnd`, account `LArTPC_FM` |

## 3. Arms

| arm | init | labels | seeds | question |
|---|---|---|---|---|
| **R1** | P3 + width (`p3/ft_width/f0.01_s{0,1,2}`) | `wc_sbnd` (all WC candidates, as E7) | 3 | P3 version of E7: how far does real data move the best model? |
| **R2** | P3 + width | `wc0_sbnd` (candidates passing the PR cosmic tagger) | 3 | does a cleaner label cut the off-beam ν charge? |
| **R3** | P3 (nominal widths, `p3/ft/f0.01_s0`) | `wc0_sbnd` | 1 | is a separate width fix still needed once real data is used? |

The seeds pair with the inits (seed k starts from init seed k). That makes 7 runs.

**Optional second stage (only if R1/R2 forget the simulation badly, ΔT1 > 0.05):**
- **R4:** mixed fine-tune, alternating batches of simulation (truth labels, data-like widths) and real data (WC labels).
- It needs a small code change (two corpora in `train_multitask`) and is not costed here.

## 4. Evaluation (no new code)

The same 13 evaluations per run as for P3 (`qlfm.eval_sbnd_data`, `report` → `report_p3wc.json`):

| metric | depends on WC labels? | what it tells |
|---|---|---|
| **Simulation T1 F1, nominal and data-like widths** (truth) | no | forgetting: how much simulation skill is lost |
| **Off-beam charge called ν** | no (off-beam has no neutrinos) | the caveat: should fall toward WC's 0.44% |
| **Beam-on pass rate at WC's off-beam rate**, and the beam-on − off-beam excess | weakly (T2 is trained on "has a WC hit") | real neutrino selection at fixed cosmic background |
| Agreement with WC ν charge on held-out beam-on (WC charge tagged / model charge in WC) | **yes: rises by construction** | report it, but do not read it as an improvement on its own |
| Agreement with cosmic-tagged WC (`wc0`) for R1, and with all-WC for R2/R3 | partly | cross-check between the two label choices |

Event displays:
- the same 40 events picked for P3 + width (`p3/displays_w/events.json`);
- R1 s0, R2 s0 and P3 + width side by side;
- shows directly what changed on each event.

## 5. What would count as success
- Off-beam ν charge drops from 1.9–3.1% toward WC's 0.44% (R2 expected to do best).
- Beam-on excess stays ≥ 35 points.
- Simulation T1 at data-like widths stays ≥ 0.93 (it is 0.965 now; E7 lost 0.06).

If R2 meets all three, it is a candidate default for the paper's real-data results. If simulation T1 drops more, go to R4.

## 6. Cost and timing
- **Training:** 7 runs × about 20–25 min ≈ 0.7 node-h, packed four per node.
- **Evaluation:** 7 × 13 evaluations × about 1.5 min ≈ 0.6 node-h.
- **Displays:** < 0.1 node-h.
- **Total ≈ 1.5 node-h on `LArTPC_FM`.** About 2–3 hours wall-clock on the debug queue (one task file, `queue.pbs`).

## 7. Caveats
- **Circularity:** on data, WC is the label and the reference. Only simulation truth and off-beam are independent of it.
- **The label inherits WC's choices:** `wc_sbnd` includes candidates the cosmic tagger flags; `wc0_sbnd` drops genuine neutrinos the tagger removes.
- **Trigger difference:** the data are light-triggered and the simulation is not, so event-level rates are not exposure-normalised.

## 8. Results (2026-11-27)

Training: job 7892241, `debug-scaling`, 7 runs about 35 min. Evaluation: 91 runs, no failures. Outputs:
- runs: `LArTPC_FM/yuhw/fm4npp-lartpc-work/p3/wcft/{R1,R2,R3}/f1_s*`;
- reports: `.../sbnd_data/eval_p3wc/report_all.jsonl`.

All data columns use the held-out files. Simulation uses the truth test set.

| arm | sim T1 F1, nominal / data-like widths | WC ν charge tagged | model charge in WC | beam-on selected (off-beam at WC's 6.0%) | off-beam charge called ν |
|---|---|---|---|---|---|
| **R1** P3 + width, `wc_sbnd` (3 seeds) | 0.875–0.890 / **0.948–0.950** | **0.831–0.833** | 0.949–0.951 | **51.3–52.1%** | **0.06–0.08%** |
| **R2** P3 + width, `wc0_sbnd` (3 seeds) | 0.854–0.858 / 0.941–0.943 | 0.799–0.803 | 0.954–0.956 | 52.4–53.2% | 0.04% |
| **R3** P3, `wc0_sbnd` (1 seed) | 0.831 / 0.940 | 0.796 | 0.957 | 53.5% | 0.04% |
| P3 + width (start point) | 0.872–0.896 / 0.965 | 0.813–0.826 | 0.889–0.899 | 41.4–41.6% | 1.9–3.0% |
| Wire-Cell | 0.93 | – | – | 44.8% | 0.44% |

**Success bars (section 5): all met by every arm.**
- **Off-beam ν charge:** 0.04–0.08%, below WC's 0.44%. Event gating no longer changes it: the hit head itself stopped tagging in-time cosmics.
- **Beam-on excess:** about 45–47 points at WC's off-beam rate (WC 39).
- **Simulation T1 at data-like widths:** 0.940–0.950. The loss is 0.015–0.025, far less than E7's 0.06.

**Reading**
- **R1 is the best overall.** Cosmic-tagged labels (R2) do not help further on any independent metric.
- **R3 ≈ R2:** once real data is used, the separate width fine-tune adds little.
- **R4 (mixed fine-tune) is not needed.**
- **Circularity:** WC agreement and event selection are trained toward WC. Simulation truth and off-beam charge are the independent checks.

Event displays of R1 / R2 / P3 + width on the same 40 events: `p3/displays_wcft` (job 7892558).
