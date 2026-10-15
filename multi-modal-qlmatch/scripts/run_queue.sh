#!/bin/bash
# Per-node worker launcher: 4 GPU workers scan TASKFILE (one `module args... --out DIR` per line).
# A task is skipped if DIR/metrics.json exists or DIR.lock is held (claimed by another worker/job),
# so several jobs can share one task file. Tasks that cannot finish before DEADLINE are left for later jobs.
# Usage (inside PBS): mpiexec -n NNODES --ppn 1 bash run_queue.sh TASKFILE DEADLINE_EPOCH
source /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch/scripts/env.sh
TASKS=$1; DEADLINE=$2
worker() {
  local gpu=$1
  while true; do
    local ran=0
    while read -r line; do
      [ -z "$line" ] && continue
      local out=$(echo "$line" | sed -E 's/.*--out ([^ ]+).*/\1/')
      [ -f $out/metrics.json ] && continue
      local left=$(( (DEADLINE - $(/bin/date +%s)) / 60 - 3 ))
      [ $left -lt 10 ] && return
      # long runs (large label budgets, encoder trained) need a fresh slot
      if [[ "$line" =~ --label-frac\ (1|0\.1)\  ]] && [[ ! "$line" =~ --mode\ (frozen|linear) ]] && [ $left -lt 40 ]; then continue; fi
      mkdir -p $(dirname $out)
      mkdir $out.lock 2>/dev/null || continue
      mkdir -p $out
      echo "[$(hostname) gpu$gpu $(/bin/date +%T)] -> $out (${left} min left)"
      CUDA_VISIBLE_DEVICES=$gpu timeout $((left*60)) $PY -m $line --time-limit $((left-5)) > $out/run.log 2>&1
      rmdir $out.lock
      ran=1
    done < $TASKS
    [ $ran -eq 0 ] && return
  done
}
for g in 0 1 2 3; do worker $g & done
wait
