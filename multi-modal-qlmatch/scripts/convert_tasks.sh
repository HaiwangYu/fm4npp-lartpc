#!/bin/bash
# Emit "INPUT OUTDIR PART NPARTS" lines for all downloaded shards.
D=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data
W=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/cache
NP_BNB=${NP_BNB:-6}
for f in $D/bnb_nowire/BNB_All_NoWire_*.h5; do
  b=$(basename $f .h5)
  for p in $(seq 0 $((NP_BNB-1))); do echo "$f $W/bnb/${b}_p$p $p $NP_BNB"; done
done
for f in $D/nue_nowire/nue_NoWire_*.h5; do
  b=$(basename $f .h5); echo "$f $W/nue/${b}_p0 0 1"
done
