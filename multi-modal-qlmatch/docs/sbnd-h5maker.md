# SBND HDF5 sample for qlfm: progress log (#11)

Goal: an SBND sample in the MicroBooNE open-data HDF5 layout (zenodo 8370883, `HDF5Maker`). It is built from:
- Avinay's Gen2 reco1 MC (BNB CV; later also νe CC and NC);
- our Wire-Cell img→PR chain output, aurora-r2 (sbnd-wirecell-production #4).

The content follows `data-inventory-sbnd.md` §4. Code: `sbnd-h5maker/` (README = table layout and methods).

Produced on ALCF Aurora:
- data: `/lus/flare/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data`;
- copied for Polaris to `/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data`.

The Polaris session `fm4npp-lartpc@polaris` checks each sample before the next step.

## Owner decisions (2026-11-12)

- **Wire-Cell content:** event level (ν candidate, flash, scores, reco vertex and energy) plus hit level (WC ν-candidate cluster footprint per gaushit). This is the SBND stand-in for the Pandora baseline.
- **MC truth:** the MicroBooNE edep layout (hit → particle energy and fraction, particle table with category), plus cosmic truth.
- **Per-hit truth:** the reco1 MC has none, so it is built by projecting the 3D Geant4 deposits (below).
- **Sample sizes:** ~10 BNB CV events first, then 10k BNB CV, then a cost estimate for ~1M (CV + νe CC + NC). Nothing larger runs before the owner approves.

## Inputs

- **Reco1:** `/lus/flare/projects/neutrinoGPU/abhat/sbnd/gen2-wct0319-bnb-1m-restartable-20261013-shard0{1..3}` (+ batch01–09).
  - sbndcode v10_14_02_03; `prodgenie_corsika_proton_rockbox0p1_sbnd`, so cosmics are simulated (CORSIKA).
  - 1–3 events per reco1 file.
- **Wire-Cell:** aurora-r2 unit directories `production-prep/dlspace/bnb/node<k>/<unit>/`, with `reco1_files.json` and `pr/all/pr_evt<r>_<s>_<e>/tracking-pr.root`.
  - 11,888 BNB units, 627,114 events with WC output.

## What the reco1 files hold

| Product | Status | Used for |
|---|---|---|
| `gaushit` | yes | `hit_table` |
| `gaushitTruthMatch` (hit → MCParticle, BackTrackerHitMatchingData) | **empty** in every event | — |
| `simtpc2d:simpleSC` SimChannels | one per channel (11,276), **no IDEs** | — |
| `ionandscint:priorSCE` SimEnergyDeposits | yes (~12 MB/event) | per-hit truth by projection |
| `largeant` MCParticles + MCParticle→MCTruth assns + ParticleAncestryMap | yes (every particle has an MCTruth) | `particle_table`, origin ν / cosmic |
| `generator` MCTruth, BeamGateInfo | yes | `event_table` |
| `mcreco` MCShower / MCTrack | yes | HDF5Maker particle categories |
| `ophitpmt`, `opflashtpc0/1` | yes; PMT readout ±1.5 ms, ~36k ophits/event | `ophit_table` (−10…40 µs window), `opflash*` (all) |

## Per-hit truth by deposit projection

Each deposit midpoint is processed as follows:
1. Shift it by the forward SCE map (`SCEoffsets_SBND_E500_dualmap_CV_voxelTH3.root`, `TrueFwd_*`), as the Wire-Cell truth labeler does.
2. Drift it to each wire plane at 0.1563 cm/µs.
3. Map it to a channel with the linear wire-plane model from the WCT wire geometry (`sbnd-wires-geometry-v0206`). Every plane is linear to < 0.02 cm, and the channel model agrees with the `(tpc, plane)` of all hits.
4. Compute tick = 410 + 2·(t_G4 + t_drift)[µs] + dt[plane].
5. Sum the deposits per particle within the hit's PeakTime ± 1 RMS, the larsim BackTracker window.

Calibration on the first 3 BNB CV events, maximising the hits that get a deposit:
- dt = +5 (U), +2 (V), −2 (Y) ticks;
- the peaks are broad (±2 ticks within 0.2%).

| 10 BNB CV events | dt = 0 | dt = (5, 2, −2), ±1 RMS | dt = (5, 2, −2), ±2 RMS |
|---|---|---|---|
| Hits with no deposit | 45,587 (30.6%) | 3,788 (2.5%) | 3,109 (2.1%) |
| ν energy on collection hits / projected ν energy (collection) | 89.9% | 91.5% | 101.5% (double counting) |
| WC ν-cluster hit purity vs truth ν hits | 0.78 | 0.87 | 0.89 |

The default is ±1 RMS.

## Status

| Date | Step | Result |
|---|---|---|
| 2026-11-12 | Converter, detector JSON, check script | Done. Debug-queue jobs 9079466–9080493 (development). |
| 2026-11-12 | 10 BNB CV events (job 9080512) | `sbnd-bnb-cv/test10/sbnd_bnb_cv_test10.h5`: 4.7 MB, `check_h5.py`: no layout problems. Copied to Eagle; sent to `fm4npp-lartpc@polaris` for checking. |

### test10 summary (`check_h5.py`)

- **Events:** 10 (one aurora-r2 unit), all `wc_status` = 1. ν: 10 νμ, 7 CC, 5 with the vertex in the AV; 4 events have 2 ν MCTruths.
- **Hits:** 148,939 (2.5k–25k per event; cosmics dominate). `local_wire` up to 1972 (U/V) and 1669 (Y); ticks 1–3425 (readout 3,427 ticks).
- **Truth:** 7,534 hits with a ν deposit > 0.05 MeV and 137,587 with a cosmic deposit; 1,059 particles (472 ν, 587 cosmic).
- **Light:** 15,113 ophits in −10…40 µs (155–2,971 per event); 97 PMT channels; beam flash at 0–4 µs. 409 flashes over the full readout.
- **Wire-Cell:** 6 ν candidates in 6 events. WC ν-cluster hits have purity 0.87 against truth ν hits. Efficiency is 0.29 over all events (5 of 10 ν vertices are outside the AV).

## Notes for the qlfm side

Differences from the MicroBooNE files:
- **Two TPCs:** `tpc` is 0 for x < 0 and 1 for x > 0, and the drift sign differs per TPC. Wires: 1,984 U / 1,984 V / 1,670 Y per TPC, 0.3 cm pitch. Readout 3,427 ticks of 0.5 µs; trigger at tick 410.
- **PMTs:** 120 PMTs, OpChannel 0–311, mixed with X-ARAPUCA channel numbers. There is no `hit_channel < 32` assumption to keep. Channel → (x, y, z, type) is in the `detector` attribute.
- **Cosmic truth:** `edep_table` keeps MicroBooNE semantics (ν only); the cosmic deposits are in `edep_cosmic_table`.
- **WC labels** replace the Pandora tables.
