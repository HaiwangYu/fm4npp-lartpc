# Round-1 curated MicroBooNE samples (about 1000 events each, with Wire-Cell ν selection)

Selected 2026-11-11 (see `docs/uboone-data-curation.md` §4 and issue #10). Tape staging was
agreed with the production team (Ben).

| Sample | Input artroot | Events | In WC checkout | WC nusel files | Input to stage | nusel to stage |
|---|---|---|---|---|---|---|
| `bnb_overlay` | 30 public files (`/pnfs/uboone/persistent/PublicAccess/...filter_run1_reco2_reco2`, on disk) | 1013 | 1009 | 30 | none | 3.8 GB |
| `nue_overlay` | 30 public files (`...intrinsice_nue..._run1_reco2_reco2`, on disk) | 1271 | 1012 | 30 | none | 6.3 GB |
| `beam_on` | 8 files of `data_bnb_mcc9.1_v08_00_00_25_reco2_C1_beam_good_reco2` | 1059 | all subruns | 83 (3 of 86 reco1 parents have none) | 5.7 GB | 3.8 GB |
| `ext` | 4 files of `data_extbnb_mcc9.1_v08_00_00_25_reco2_C2_all_reco2` (runs 7535, 7640, 7712, 7762) | 1248 | 1248 | 46 | 6.5 GB | 4.5 GB |

Selection:
- **MC:** public files ranked by the fraction of their events found (by run, subrun, event) in
  `checkout_prodgenie_bnb_{nu,intrinsic_nue}_overlay_run1_PF.root`, taken until about 1000
  shared events. Event IDs came from `scripts/artroot_event_ids.py` on random samples of 120
  (ν) and 60 (ν_e) files.
- **Beam-on:** random C1 beam-good reco2 files with every subrun in the WC
  `checkout_data_bnb_run1_PF.root` `T_pot`.
- **EXT:** files whose events are at least 99% covered by WC `checkout_data_extbnb_run1_PF.root`
  (WC event counts per subrun), 120-320 events each. Only C2 files passed in the sampled range.
- **WC nusel files:** for each input file, its samweb reco1 parents (skipping the `reco2_*`-stage
  intermediates), then each parent's `nusel*_WCP_*` child from the WC production
  (`prodgenie_bnb_{nu,intrinsic_nue}_uboone_overlay_mcc9.0_run1_wcp_reco1.5`,
  `data_{bnb,extbnb}_mcc9.1_wcp_port_TEST`).

Files: `<sample>_inputs.txt` (artroot), `<sample>_nusel.txt` (WC outputs), `selection.json`
(everything, with per-file event counts and lineage), `to_stage.txt` (the 201 tape files: data
reco2 plus all nusel, about 30.6 GB).
