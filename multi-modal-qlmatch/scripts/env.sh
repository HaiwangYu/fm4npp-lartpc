# Source me: Polaris environment for qlfm
module use /soft/modulefiles >/dev/null 2>&1
module load conda/2025-09-25 >/dev/null 2>&1
export PY=/soft/applications/conda/2025-09-25/mconda3/bin/python
export QLFM_ROOT=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc/multi-modal-qlmatch
export QLFM_DATA=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-data
export QLFM_WORK=/lus/eagle/projects/neutrinoGPU/yuhw/fm4npp-lartpc-work
export PYTHONPATH=$QLFM_ROOT:$PYTHONPATH
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TMPDIR=/tmp   # mpiexec sets a long TMPDIR -> DataLoader "AF_UNIX path too long" hang
