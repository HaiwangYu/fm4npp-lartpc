# multi-modal-qlmatch — charge + light self-supervised foundation model for LArTPC (MicroBooNE open data)

Prototype of a multi-modal (TPC charge hits + PMT light) foundation model pretrained with masked
autoencoding, evaluated with small downstream heads (ν/cosmic at hit and event level, flavor, energy).
See `docs/plan-initial-round.md` for the design and experiment plan.

## Layout
```
qlfm/geometry.py            MicroBooNE constants, PMT map
qlfm/data/h5reader.py       open-data HDF5 reader (seq_cnt indexing)
qlfm/data/convert.py        HDF5 shard -> column-wise .npy cache (+ per-hit nu truth, Pandora baseline)
qlfm/data/dataset.py        memmap dataset + packed collate
qlfm/data/splits.py         (run, subrun)-hashed splits, nested subsets
qlfm/data/labels.py         downstream task definitions (single source of truth)
qlfm/tokenize.py            GPU tokenizers (wire-time cell tokens; per-PMT time-profile tokens)
qlfm/models/encoder.py      embedders + FlashAttention varlen Transformer
qlfm/models/fm.py           FM backbone + multi-modal MAE
qlfm/models/heads.py        attentive-pool / linear / per-hit heads
qlfm/train_pretrain.py      DDP pretraining (resumable, time-limited)
qlfm/train_downstream.py    scratch | frozen | finetune | linear runs
qlfm/evaluate.py            metrics
scripts/                    download, conversion, splits, task generation, queue runner, summaries, PBS
```

## Quick start (Polaris)
```bash
source scripts/env.sh
python scripts/download_zenodo.py 8370883 $QLFM_DATA/bnb_nowire --jobs 18      # inclusive NoWire
qsub scripts/pbs/convert.pbs                                                    # -> $QLFM_WORK/cache/{bnb,nue}
$PY scripts/make_splits.py $QLFM_WORK/cache/bnb $QLFM_WORK/cache/splits_bnb.npz
mpiexec -n 8 --ppn 4 $PY -m qlfm.train_pretrain --out $QLFM_WORK/pre/cl_f0.90 --light 1
$PY scripts/make_tasks.py e3 --fm-c .../fm.pt --fm-cl .../fm.pt --out-root $QLFM_WORK/e3 > tasks.txt
qsub -q debug -l select=2:system=polaris -l walltime=01:00:00 -v TASKS=$PWD/tasks.txt scripts/pbs/queue.pbs
$PY scripts/summarize.py $QLFM_WORK/e3
```
## Polaris notes
- `debug`: 1 running + 1 queued job per user (≤2 nodes); `debug-scaling`: 1 job total, running or queued (≤10 nodes).
  Pack work: `scripts/pbs/pretrain_multi.pbs` (several pretraining runs per job, `NODES_PER_RUN`) and
  `scripts/pbs/queue.pbs` + `scripts/run_queue.sh` (single-GPU downstream runs, 4 per node).
- No ssh to compute nodes: `scripts/pbs/hold.pbs` runs `scripts/node_agent.sh`; send commands with
  `scripts/box_run.sh NAME TIMEOUT <<'EOF' ... EOF`.
- `scripts/env.sh` sets `TMPDIR=/tmp` (mpiexec's long TMPDIR breaks DataLoader workers).
- Running log of decisions/findings: `docs/round1-log.md`.
- Round-1 tech note: `docs/technote-round1.md` (Markdown) and `docs/technote-round1.pdf`
  (LaTeX source in `docs/technote-round1-tex/`; build with Tectonic:
  `tectonic -X compile docs/technote-round1-tex/technote-round1.tex`, a user-space binary is in
  `$QLFM_WORK/tools/tectonic/`).

Data: MicroBooNE open samples (CC BY 4.0), Zenodo 10.5281/zenodo.8370883 and 10.5281/zenodo.7261921.
