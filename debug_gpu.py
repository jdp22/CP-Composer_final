import torch
import torch.distributed as dist

def test_nccl():
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        rank=0,
        world_size=1
    )
    print("NCCL initialized successfully!")

test_nccl()