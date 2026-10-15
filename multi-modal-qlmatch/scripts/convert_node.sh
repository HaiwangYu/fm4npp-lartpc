#!/bin/bash
# Run on each node: convert the subset of tasks assigned to this node (round-robin), NPROC at a time.
source /lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch/scripts/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
NODE=${PMI_RANK:-0}; NNODES=${PMI_SIZE:-1}; NPROC=${NPROC:-28}
bash $QLFM_ROOT/scripts/convert_tasks.sh | awk -v n=$NNODES -v r=$NODE 'NR%n==r' | \
  xargs -P $NPROC -L 1 bash -c '[ -f $1/meta.json ] || { mkdir -p $1; $PY -m qlfm.data.convert $0 $1 --part $2 --nparts $3 > $1.log 2>&1; echo "done $1 $?"; }'
