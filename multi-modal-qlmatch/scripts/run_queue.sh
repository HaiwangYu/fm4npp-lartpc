#!/bin/bash
# Per-node worker launcher: 4 GPU workers pull lines (one command each, last arg chunk "--out DIR")
# from TASKFILE using a flock-protected counter; tasks whose DIR/metrics.json exists are skipped.
# Usage (inside PBS): mpiexec -n NNODES --ppn 1 bash run_queue.sh TASKFILE DEADLINE_EPOCH
source /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch/scripts/env.sh
TASKS=$1; DEADLINE=$2
CNT=$TASKS.counter; LOCK=$TASKS.lock
[ -f $CNT ] || echo 1 > $CNT
worker() {
  local gpu=$1
  while true; do
    local n=$(flock $LOCK bash -c "n=\$(cat $CNT); echo \$((n+1)) > $CNT; echo \$n")
    local line=$(sed -n "${n}p" $TASKS)
    [ -z "$line" ] && break
    local out=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/')
    [ -f $out/metrics.json ] && continue
    local left=$(( (DEADLINE - $(date +%s)) / 60 - 3 ))
    if [ $left -lt 8 ]; then echo "$line" >> $TASKS.unfinished; continue; fi
    mkdir -p $out
    echo "[$(hostname) gpu$gpu] task $n -> $out (${left} min left)"
    CUDA_VISIBLE_DEVICES=$gpu timeout $((left*60)) $PY -m $line --time-limit $((left-4)) > $out/run.log 2>&1
    [ -f $out/metrics.json ] || echo "$line" >> $TASKS.unfinished
  done
}
for g in 0 1 2 3; do worker $g & done
wait
