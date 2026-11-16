import json
import platform
import sys

import geoopt.manifolds.siegel.csym_math as sm
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
results = []
for batch, size in [(16, 32), (4, 128), (1, 256)]:
    torch.manual_seed(1992)
    raw = torch.randn(batch, size, size, dtype=torch.complex128)
    symmetric = (raw + raw.transpose(-1, -2)) * 0.5
    bounded = symmetric / (2 * torch.linalg.matrix_norm(symmetric, keepdim=True))
    y = symmetric.imag @ symmetric.imag.transpose(-1, -2) / size + torch.eye(size)
    upper = torch.complex(symmetric.real, y)
    weight = torch.randn_like(raw)
    for name, source, function in [
        ("cayley", bounded, sm.cayley_transform),
        ("inverse_cayley", upper, sm.inverse_cayley_transform),
    ]:
        for backward in [False, True]:
            point = source.detach().requires_grad_(backward)

            def run(function=function, point=point, backward=backward, weight=weight):
                result = function(point)
                if backward:
                    torch.autograd.grad((result * weight).real.sum(), point)

            measurement = Timer(
                stmt="run()", globals={"run": run}, num_threads=1
            ).blocked_autorange(min_run_time=0.3)
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                run()
            record = {
                "name": name,
                "backward": backward,
                "batch": batch,
                "size": size,
                "median_ms": measurement.median * 1000,
                "iqr_ms": measurement.iqr * 1000,
                "inversion_calls": sum(
                    event.name == "aten::linalg_inv" for event in prof.events()
                ),
            }
            results.append(record)
            print(json.dumps(record), flush=True)

with open(sys.argv[1], "w") as output:
    json.dump(
        {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "system": platform.platform(),
            "threads": 1,
            "dtype": "complex128",
            "results": results,
        },
        output,
        indent=2,
    )
