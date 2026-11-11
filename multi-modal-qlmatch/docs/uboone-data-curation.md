# MicroBooNE data curation: input tier, Wire-Cell overlap, real-data inputs

Status 2026-11-11. Tracking issue: [#10](https://github.com/HaiwangYu/fm4npp-lartpc/issues/10).

So far the project has trained on the MicroBooNE open-data HDF5 (Zenodo
[8370883](https://zenodo.org/records/8370883), inclusive BNB overlay, NoWire; and
[7261921](https://zenodo.org/records/7261921), the nue sample). This note covers three questions:

1. From which data tier can HDF5 like ours be made?
2. Does the open dataset overlap the existing Wire-Cell (WC) final outputs ("checkout" files),
   so the two can be compared event by event?
3. For real data (BNB beam-on and beam-off EXT), where are the input files (samweb)?

Josh's setup for rebuilding `hdf5maker` and converting the public artroot on the grid is in
[JoshHenryPeterson/fm4npp-lartpc: multi-modal-qlmatch/docs/microboone_artroot_to_hdf5.md](https://github.com/JoshHenryPeterson/fm4npp-lartpc/blob/main/multi-modal-qlmatch/docs/microboone_artroot_to_hdf5.md).
This note does not repeat it.

Tools used: `samweb` from CVMFS
(`/cvmfs/fermilab.opensciencegrid.org/products/common/prd/sam_web_client/v3_6/NULL/bin/samweb -e uboone`,
read-only queries need no proxy), `uproot`, and `scripts/artroot_event_ids.py`, which dumps
(run, subrun, event) from artroot without art.

## 1. Input tier for the HDF5

**Answer: reco2 artroot.** The public HDF5 was made by the `HDF5Maker` analyzer
([uboone/hdf5maker](https://github.com/uboone/hdf5maker), branch `opensamples`) run on the
MCC9.1 run1 overlay reco2 files. MCC9 reco2 files keep the reco1 products, including the
`butcher` wires, so one reco2 file holds every product HDF5Maker reads.

The public artroot is on dCache disk (readable without credentials over xrootd, and directly
on the GPVMs):

| Directory under `/pnfs/uboone/persistent/PublicAccess/` | Files |
|---|---|
| `prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2` | 24,436 |
| `prodgenie_bnb_nu_uboone_overlay_mcc9.1_v08_00_00_26_filter_run1_reco2_reco2_nfiles3400` | 3,400 |
| `prodgenie_bnb_intrinsice_nue_uboone_overlay_mcc9.1_v08_00_00_26_run1_reco2_reco2` | 2,151 |

The samweb definition with the same name as the first directory has 24,419 files, 17 fewer
than the public list. The difference was not checked.

Products HDF5Maker reads (labels from `HDF5Maker.fcl`), as found in one public file:

| HDF5Maker input | Product in the reco2 file | Producing art process (stage) |
|---|---|---|
| HitLabel `gaushit` | `recob::Hits_gaushit` | `OverlayStage1a` (reco1) |
| WireLabel `butcher` (WithWire only) | `recob::Wires_butcher` | `OverlayStage1a` (reco1) |
| OpHitLabel `ophitBeam` | `recob::OpHits_ophitBeam` | `OverlayStage1a` and `OverlayStage1Optical` |
| OpFlashLabel `simpleFlashBeam` | `recob::OpFlashs_simpleFlashBeam` | `OverlayStage1a` and `OverlayStage1Optical` |
| HitTruthLabel `gaushitTruthMatch` | hit-MCParticle assns | `OverlayRecoStage1b` |
| PandoraLabel `pandora` | PFParticles, slices, clusters | `OverlayRecoStage2` (reco2) |
| T0Label `flashmatch` | `anab::T0s_flashmatch` | `OverlayRecoStage2` (reco2) |
| TruthLabel `generator`, `largeant`, `mcreco` | MCTruth, MCParticles, MCShower/MCTrack | gen / g4 |

Notes:
- `ophitBeam` and `simpleFlashBeam` exist from two processes. `getByLabel` without a process
  name returns the most recent one. Which one that is has not been checked, but it matters
  for the light inputs.
- reco1 files alone would lack `pandora` and `flashmatch`. HDF5Maker only needs those for the
  Pandora tables and T0, so the core hit/flash tables could be made from reco1.
- Open: the public HDF5 has 753,467 events, against 1,046,139 in the artroot (72%). The reason
  is not documented. Comparing the HDF5 `event_table` IDs with a full artroot ID dump would
  show which events were dropped (see §4).

## 2. Overlap with the Wire-Cell checkout files

### 2.1 What the checkout files contain
`/pnfs/uboone/persistent/users/jjo/processed_chekcout_rootfiles/` has `bnb/{run1,run2,run3,run4,run4a-d,run5}/{cv,data,systs}`
and `numi*`. Each file has one directory, `wcpselection`, with the trees
`T_eval`, `T_PFeval`, `T_BDTvars`, `T_KINEvars`, `T_pot`.
- `T_eval` has `run/subrun/event`, flash/match info, the `stm_*` cosmic tagging, and for MC
  the truth (`truth_nuEnergy`, `truth_vtx*`, `truth_isCC`, ...) and weights.
- `T_pot` has one entry per subrun (`runNo`, `subRunNo`, POT, spills).
- The files are not skimmed: `match_found` is true for 78.5% of the events in the run1 ν
  overlay, and events failing preselection are kept.
- **They hold no input-level information** (no hits, wires or flashes, and no input file names).
  The inputs have to be traced by event ID and samweb. For data, the reco2 file names end in
  `..._reco1_postwc_...`, so the reco2 and the WC chain share the same reco1 parents. The
  celltree definitions are in samweb (e.g. `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_celltree`,
  `data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_celltree`; `..._reco2_C1_celltree` has 0 files).

### 2.2 Open-data overlay vs WC run1 ν overlay (event level)
I dumped event IDs from a random 400 of the 24,436 public artroot files (15,709 events, all
IDs unique) and matched (run, subrun, event) against `bnb/run1/cv`:

| WC file | WC events | Public events found in WC |
|---|---|---|
| `checkout_prodgenie_bnb_nu_overlay_run1.root` | 581,941 | **52.7%** |
| `checkout_prodgenie_bnb_nu_overlay_run1_PF.root` | 581,941 | 52.7% (same events) |
| `checkout_prodgenie_bnb_nu_overlay_LowE_run1.root` | 20,364 | 0% |

The overlap is by whole subruns:
- Of the 2,036 sampled subruns, 1,055 are in the WC `T_pot`. Of those, 1,026 have every event
  matched and 29 have most of them.
- No subrun outside `T_pot` has any match.
- Where checked, the event count per subrun is identical on both sides.

So these are the same events, not chance ID collisions. Scaling 52.7% to the 1.05M public events
gives roughly 550k events in common. That would be most of the 582k WC events, but the full
count has not been made (§4).

The overlay event IDs carry the run number of the overlaid EXT data (e.g. run 7419). The
public file names start with the EXT file name.

### 2.3 Real data vs WC run1 data (subrun level)
WC run1 data checkout files (`bnb/run1/data`):

| File | Events | Runs | Subruns in `T_pot` |
|---|---|---|---|
| `checkout_data_bnb_run1_PF.root` | 544,769 | 4952-6998 | 85,064 |
| `checkout_data_bnb_run1_all.root` | 545,083 | 4952-6998 | 85,111 |
| `checkout_data_extbnb_run1_PF.root` | 633,290 | 4952-7770 | 91,566 |
| `checkout_data_bnb_run1_5e19.root` | 177,957 | 5121-5946 | 22,750 |
| `checkout_data_bnb_run1_opendata_Aug8.root` | 139,054 | 5121-5946 | 19,332 |

`checkout_data_bnb_run1_opendata_Aug8.root` is BNB beam-on data from the same run range as the
5e19 sample. It is most likely the unblinded ("open") run1 subset, **not** the Zenodo public
release, which is MC overlay only. The POT branches in the data `T_pot` are not filled
(data POT comes from the beam database).

For 40 random files of each samweb reco2 definition, the share of their subruns found in the WC
`T_pot`:

| samweb definition | Subruns in WC |
|---|---|
| `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2` vs `checkout_data_bnb_run1_PF` | 564/712 (79%) |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_all_reco2` vs `checkout_data_extbnb_run1_PF` | 1280/2208 (58%) |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C2_all_reco2` vs `checkout_data_extbnb_run1_PF` | 955/1923 (50%) |

For real data, (run, subrun, event) identifies an event uniquely, so a direct WC comparison is
possible for any reco2 data we convert. Event-level matching needs IDs read from the reco2
files, which are on tape (§3).

## 3. Inputs for new real-data HDF5

### 3.1 samweb definitions (MCC9.1, `v08_00_00_25`, reco2)
Beam-on: `data_bnb_mcc9.1_v08_00_00_25_reco2_<epoch>_beam_good_reco2`.
Beam-off EXT: `data_extbnb_mcc9.1_v08_00_00_25_reco2_<epoch>_all_reco2`.
The epochs are C1, C2, D1, D2, E1, E2, F, G1, G2, G2a. C1 and C2 are run1, based on the run
ranges above. The mapping of the other epochs to runs 2 and 3 still needs confirming.

| Definition | Files | Events | Size |
|---|---|---|---|
| `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2` | 5,864 | 638,039 | 3.44 TB |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C1_all_reco2` | 1,905 | 441,403 | 2.38 TB |
| `data_extbnb_mcc9.1_v08_00_00_25_reco2_C2_all_reco2` | 1,138 | 398,046 | 2.05 TB |

The beam-good definition is `file_type data and ub_project.name data_bnb_mcc9.1_v08_00_00_25 and data_tier reconstructed and ub_project.stage reco2_C1 and data_stream beam_good and defname: goodruns_mcc9_run1_hardcoded`,
so the run1 good-run list is already applied. File names show the chain
`..._bnb_..._merged_..._optfilter_..._reco1_postwc_...` (software optical filter, reco1, WC stage).

Location: tape only (`enstore:/pnfs/uboone/data/uboone/reconstructed/prod_v08_00_00_25/...`).
The files need to be prestaged (`samweb prestage-dataset --defname=...`) or read through a SAM
project in grid jobs. Prestaging 3-6 TB per definition needs a word with the data-management
people first.

Other related definitions exist (e.g. `data_bnb_optfilter_C1_5e19_goodruns_mcc9.1_reco2`,
`_high_lifetime`/`_low_lifetime` splits, filtered streams such as `_nucc_reco2` and
`_pi0_reco2`, `_celltree`). List them with
`samweb -e uboone list-definitions --defname='data_bnb_mcc9.1_v08_00_00_25_reco2_%'`.

### 3.2 HDF5Maker changes needed for data
`HDF5Maker_module.cc` (opensamples) assumes MC:
- With `EventInfo: "nu"` it throws if there is no MCTruth. Use `EventInfo: "none"`.
- It always reads `mcreco` MCShower/MCTrack (`getValidHandle`, which throws on data), and it
  always builds the backtracker / `ParticleInventoryService` (`initBacktrackingParticleVec`,
  `pi->Rebuild(e)`). These need to be skipped when `e.isRealData()`.
- Set `HitTruthLabel: ""` and `UseMap: false`. Check what the energy-deposit and particle
  tables do with no truth.
- `hdf5maker_uB_public_job.fcl` sets `DetectorClocksService.TrigModuleName: "triggersim"`;
  for data this should be `"daq"`. Check the remaining data-specific services, such as
  space-charge and calibration settings.
- The build is against uboonecode `v08_00_00_54` while data reco2 is `v08_00_00_25`. Reading
  should be fine, but test it on one file.

A data job fcl plus a small `isRealData()` guard in the module should be enough. Test it on one
prestaged C1 beam-good file and one EXT file.

## 4. Open items
- [ ] Full event-ID dump of all 24,436 public artroot files (about 5-6 h with 4 workers on a
      GPVM, or as grid jobs): exact count of events shared with WC run1 overlay.
- [ ] Event IDs from the open-data HDF5 `event_table` (on Polaris): why the HDF5 has 72% of the
      artroot events, and how many HDF5 events are in WC.
- [ ] Which process's `ophitBeam`/`simpleFlashBeam` HDF5Maker picked up.
- [ ] Confirm the epoch-to-run mapping (D*/E* = run2, F/G* = run3?) and pick the real-data
      sample size (all of run1? a subset with matching WC events?).
- [ ] Prestage one C1 beam-good and one C1 EXT reco2 file; check the products exist (`gaushit`,
      `butcher`, `ophitBeam`, `simpleFlashBeam`, `pandora`, `flashmatch`); patch and run HDF5Maker
      on them.
- [ ] POT / trigger counts for beam-on and EXT normalization if needed.
- [ ] Clarify "light triggered beam off": is it the standard EXT stream (external trigger plus
      the software PMT trigger / optical filter), or something else?
