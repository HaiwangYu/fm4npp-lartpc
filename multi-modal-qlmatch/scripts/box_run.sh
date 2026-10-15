#!/bin/bash
# Usage: box_run.sh NAME TIMEOUT_S <<'CMD' ... CMD   (runs on the held node, prints the log)
BOX=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work/jobs/box
n=$1; to=${2:-600}
rm -f $BOX/out/$n.log $BOX/out/$n.done
{ echo "source /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch/scripts/env.sh; cd \$QLFM_ROOT"; cat; } > $BOX/inbox/.$n.tmp
mv $BOX/inbox/.$n.tmp $BOX/inbox/$n.sh
t=0; while [ ! -f $BOX/out/$n.done ] && [ $t -lt $to ]; do sleep 3; t=$((t+3)); done
grep -v -E "bind: warning" $BOX/out/$n.log 2>/dev/null | tail -${TAILN:-60}
[ -f $BOX/out/$n.done ] && echo "[exit $(cat $BOX/out/$n.done)]" || echo "[still running after ${to}s]"
