import os, torch
import torch.distributed as dist


def init_parallel_env(rank, nprocs):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=nprocs)


def destroy_parallel_env():
    dist.destroy_process_group()


def get_device_name_and_id_list(rank):
    platform = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = f'{platform}:{rank}' if platform == 'cuda' else platform
    device_id_list = [rank] if platform == 'cuda' else None
    return device_name, device_id_list