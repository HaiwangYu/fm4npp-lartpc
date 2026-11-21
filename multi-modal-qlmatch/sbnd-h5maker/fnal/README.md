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
