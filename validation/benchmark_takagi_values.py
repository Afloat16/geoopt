import json
import platform
import sys

import geoopt
import torch
from geoopt.manifolds.siegel import csym_math as sm
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
results = []
for operation, batch, size in [
    ("takagi_values", 16, 32),
    ("takagi_values", 4, 128),
    ("takagi_values", 1, 256),
    ("bounded_distance", 4, 64),
    ("bounded_distance", 1, 128),
]:
    torch.manual_seed(1802)
    raw = torch.randn(batch, size, size, dtype=torch.complex128)
    other = torch.randn_like(raw)
    for backward in [False, True]:
        value = raw.detach().requires_grad_(backward)

        def run(value=value, other=other, backward=backward, operation=operation):
            z = (value + value.transpose(-1, -2)) / 2
            if operation == "takagi_values":
                result = sm.takagi_eigvals(z)
            else:
                y = (other + other.transpose(-1, -2)) / 2
                z = z / (2 * torch.linalg.matrix_norm(z, keepdim=True))
                y = y / (2 * torch.linalg.matrix_norm(y, keepdim=True))
                result = geoopt.BoundedDomain().dist(z, y)
            if backward:
                torch.autograd.grad(result.sum(), value)

        timing = Timer(
            stmt="run()", globals={"run": run}, num_threads=1
        ).blocked_autorange(min_run_time=0.5)
        row = {
            "operation": operation,
            "batch": batch,
            "size": size,
            "backward": backward,
            "median_ms": timing.median * 1000,
            "iqr_ms": timing.iqr * 1000,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
with open(sys.argv[1], "w") as output:
    json.dump(
        {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "threads": 1,
            "dtype": "complex128",
            "results": results,
        },
        output,
        indent=2,
    )
