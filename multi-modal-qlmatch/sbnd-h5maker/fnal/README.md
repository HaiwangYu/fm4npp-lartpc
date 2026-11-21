# sbnd-h5maker at FNAL (fm4npp-lartpc #14, porting_sbndbuild03)

Real-data HDF5 from FermiGrid inputs, option B (Wire-Cell labels from the WCLocal2026Nov16 `tracking-pr` tars, WCT 0.38.0).

- `env-fnal.sh`: FNAL equivalent of `../env.sh`. Source it inside the SL7 image (`/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest`).
  - Sets up `sbndcode v10_14_02_03` and `gallery v1_22_06` from CVMFS, plus the doc §4 wheels unpacked into `$SBNDH5_SITE`. numpy comes from UPS (1.24.3).
- `h5pair.sh <reco1> <id.tracking-pr.tar> <out.h5> [max events]`: one pair, local files only. Steps:
  1. FrameShift (`aurora/bin/step0-frameshift.sh` of sbnd-wirecell-production);
  2. untar the tracking-pr tar;
  3. `sbnd_h5maker.py --data`;
  4. `check_h5.py`.
- `copy-trpr.sh`: copies one tracking-pr tar from scratch to `/pnfs/sbnd/persistent/users/yuhw/prod-fermigrid/WCLocal2026Nov16/<sample>/<id>/`. It checks ADLER32 on both sides.
- `run-test.sh`: the test10 run on sbndbuild03, inside the SL7 image.
- `h5job.sh <list> <pairs per job> <output base>`: the FermiGrid job.
  - Uses `ifdh cp` for the reco1 and the tar, with a size check.
  - Runs `h5pair.sh` on all events.
  - Puts `sbnd_<sample>_<id>.h5`, `.h5.stats.json`, `.check.json` and `logs/<id>.logs.tar.gz` under `<output base>/<sample>/`, then writes `<id>.done` last. The `.done` file holds the status and the size and sha256 of each file.
  - A pair with a `.done` is skipped.
  - The code tarball is `h5/{maker,site,scripts,PROVENANCE.txt}`. It is shipped with `--tar_file_name dropbox://`, and the SL7 image is `fnal-dev-sl7`.
- Grid test 2026-11-20, cluster 87546314, one pair each: OK, `check_h5` clean.
  - About 14 MB and 1–1.5 min per 50-event pair.
  - The first try (94508500) failed on bash 4.2's unbound empty array under `set -u`, now fixed.
