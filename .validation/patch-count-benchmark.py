import argparse
import hashlib
from pathlib import Path
import sys

import torch

import kornia

root = Path.cwd().resolve()
assert Path(kornia.__file__).resolve().parent == root / "kornia", kornia.__file__
print("kornia:", kornia.__file__, "python:", sys.executable, flush=True)
sys.path.insert(0, str(root / "benchmarks"))
from common import run_metadata, save_json, time_us, warm_up_cpu

parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True)
parser.add_argument("--only", default="")
args = parser.parse_args()
torch.manual_seed(0)
torch.set_num_threads(1)
warm_up_cpu()
results = []
workloads = [
    ("gray", (1, 1, 256, 256), 32, 16),
    ("rgb", (1, 3, 512, 512), 32, 16),
    ("rgb_batch", (8, 3, 256, 256), 32, 16),
    ("features", (2, 32, 128, 128), 16, 8),
    ("features_batch", (4, 64, 128, 128), 16, 8),
]
for name, shape, window, stride in workloads:
    if args.only and name not in args.only.split(","):
        continue
    image = torch.rand(shape)
    patches = kornia.contrib.extract_tensor_patches(image, window, stride).detach()
    patches.requires_grad_(True)

    def combine():
        return kornia.contrib.combine_tensor_patches(patches, shape[-2:], window, stride)

    def forward_backward():
        output = combine()
        return torch.autograd.grad(output.square().mean(), patches)[0]

    output = combine()
    gradient = forward_backward()
    torch.testing.assert_close(output, image)
    record = {
        "name": name,
        "shape": list(shape),
        "window": window,
        "stride": stride,
        "dtype": str(patches.dtype),
        "output_sha256": hashlib.sha256(output.detach().numpy().tobytes()).hexdigest(),
        "gradient_sha256": hashlib.sha256(gradient.detach().numpy().tobytes()).hexdigest(),
    }
    for mode, fn in (("forward", combine), ("forward_backward", forward_backward)):
        median, iqr = time_us(fn, min_run_time=1.0)
        record[mode + "_median_us"] = median
        record[mode + "_iqr_us"] = iqr
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        probe = combine()
    record["forward_total_positive_tensor_allocation_bytes"] = sum(
        max(0, event.self_cpu_memory_usage) for event in prof.events()
    )
    results.append(record)
    print(record, flush=True)
metadata = run_metadata(torch.device("cpu"))
metadata.update({
    "kornia_module": str(Path(kornia.__file__).resolve().relative_to(root)),
    "interpreter": sys.executable,
    "source_sha256": hashlib.sha256((root / "kornia/contrib/extract_patches.py").read_bytes()).hexdigest(),
    "memory_metric": "Sum of positive self_cpu_memory_usage events for one public forward; total tensor allocation, not peak RSS.",
})
save_json(args.json, metadata, results)
