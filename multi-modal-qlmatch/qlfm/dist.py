"""Minimal torch.distributed setup for mpiexec launches on Polaris (PALS/PMI env)."""
import os

import torch
import torch.distributed as dist


def init():
    rank = int(os.environ.get("PMI_RANK", os.environ.get("RANK", 0)))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("PMI_LOCAL_RANK", os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        if "MASTER_ADDR" not in os.environ:
            nodefile = os.environ.get("PBS_NODEFILE")
            os.environ["MASTER_ADDR"] = open(nodefile).readline().strip() if nodefile else "localhost"
        os.environ.setdefault("MASTER_PORT", "29511")
        torch.cuda.set_device(local % torch.cuda.device_count())
        dist.init_process_group("nccl", rank=rank, world_size=world,
                                device_id=torch.device("cuda", local % torch.cuda.device_count()))
    elif torch.cuda.is_available():
        torch.cuda.set_device(local % torch.cuda.device_count())
    return rank, world, local


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def all_reduce_mean(t):
    if dist.is_initialized():
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return t


def barrier():
    if dist.is_initialized():
        dist.barrier()
