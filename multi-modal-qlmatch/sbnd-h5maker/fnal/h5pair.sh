#!/bin/bash
# fm4npp #14: one real-data pair -> sbnd-h5maker HDF5 (porting_sbndbuild03). Run INSIDE the SL7 image
# with scripts/env-fnal.sh sourced. Local files only (the caller fetches and stores):
#   h5pair.sh <reco1.root> <id.tracking-pr.tar> <out.h5> [max events]
# Steps (doc multi-modal-qlmatch/docs/sbnd-realdata.md §4): FrameShift (step0-frameshift.sh, sbndcode
# run_frameshift.fcl), untar the tracking-pr tar, sbnd_h5maker.py --data, check_h5.py.
# Writes <out.h5>, <out.h5>.stats.json, <out%.h5>.check.json and logs in $(dirname out)/logs/.
set -u
R1=${1:?reco1}; TAR=${2:?tracking-pr tar}; OUT=${3:?out.h5}; MAXEV=${4:-0}
H5M=${SBNDH5_MAKER:?dir with sbnd_h5maker.py}; STEP0=${SBNDH5_STEP0:?step0-frameshift.sh}
D=$(cd "$(dirname "$OUT")" && pwd); B=$(basename "$OUT" .h5); L=$D/logs; mkdir -p "$L"; W=$D/work-$B; rm -rf "$W"; mkdir -p "$W"
t0=$(date +%s)
bash "$STEP0" "$R1" "$W/fs/$(basename "$R1" .root)_frameshift.root" > "$L/$B.frameshift.out" 2>&1 || { echo "FAIL frameshift rc $?"; cp "$W"/fs/frameshift-lar.log.gz "$L/$B.frameshift-lar.log.gz" 2>/dev/null; exit 11; }
cp "$W"/fs/frameshift-lar.log.gz "$L/$B.frameshift-lar.log.gz"; t1=$(date +%s)
mkdir -p "$W/wc" && tar -C "$W/wc" -xf "$TAR" || { echo "FAIL untar"; exit 12; }
nwc=$(ls -d "$W"/wc/pr/all/pr_evt* 2>/dev/null | wc -l)
mx=(); [ "$MAXEV" -gt 0 ] && mx=(--max-events "$MAXEV")
python3 "$H5M/sbnd_h5maker.py" "$OUT" --data --reco1 "$W/fs/$(basename "$R1" .root)_frameshift.root" --wc-dir "$W/wc/pr/all" "${mx[@]}" > "$L/$B.h5maker.log" 2>&1 || { echo "FAIL h5maker rc $?"; tail -5 "$L/$B.h5maker.log"; exit 13; }
t2=$(date +%s)
python3 "$H5M/check_h5.py" "$OUT" --json "$D/$B.check.json" > "$L/$B.check.log" 2>&1; crc=$?
echo "OK h5pair $B: frameshift $((t1-t0)) s, h5maker $((t2-t1)) s, wc dirs $nwc, check rc $crc"
rm -rf "$W"
exit $([ $crc = 0 ] && echo 0 || echo 14)
