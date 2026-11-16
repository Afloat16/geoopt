from pathlib import Path
import sys
import tempfile
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.multiprocessing as mp
from sentence_transformers.util.distributed import all_gather

def worker(rank, path):
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for grad in [False, True]:
            base = (torch.arange(12.) + rank * 12).reshape(2, 6).requires_grad_()
            tensor = base[:, ::2]
            out = all_gather(tensor, with_grad=grad)
            expected = torch.cat([torch.arange(12.).reshape(2, 6)[:, ::2] + r * 12 for r in range(2)])
            print(rank, grad, tensor.stride(), out, flush=True)
            torch.testing.assert_close(out, expected)
            out.sum().backward()
            print('gradient', rank, grad, base.grad, flush=True)
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        mp.spawn(worker, args=(str(Path(directory).resolve() / 'group'),), nprocs=2, join=True)
