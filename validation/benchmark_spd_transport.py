import json
import platform
import sys

import geoopt
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
results = []
manifold = geoopt.SymmetricPositiveDefinite()
for batch, size in [(16, 32), (4, 128), (1, 256)]:
    torch.manual_seed(1732)
    raw_x = torch.randn(batch, size, size, dtype=torch.float64)
    raw_y = torch.randn_like(raw_x)
    x_base = raw_x @ raw_x.transpose(-1, -2) / size + torch.eye(size)
    y_base = raw_y @ raw_y.transpose(-1, -2) / size + torch.eye(size)
    raw_v = torch.randn_like(raw_x)
    v_base = (raw_v + raw_v.transpose(-1, -2)) / 2
    weight = torch.randn_like(raw_x)
    for backward in [False, True]:
        x = x_base.detach().requires_grad_(backward)
        y = y_base.detach().requires_grad_(backward)
        v = v_base.detach().requires_grad_(backward)

        def run(x=x, y=y, v=v, backward=backward, weight=weight):
            result = manifold.transp(x, y, v)
            if backward:
                torch.autograd.grad((result * weight).sum(), (x, y, v))

        measurement = Timer(
            stmt="run()", globals={"run": run}, num_threads=1
        ).blocked_autorange(min_run_time=0.4)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run()
        record = {
            "backward": backward,
            "batch": batch,
            "size": size,
            "median_ms": measurement.median * 1000,
            "iqr_ms": measurement.iqr * 1000,
            "eigh_calls": sum(
                event.name == "aten::linalg_eigh" for event in prof.events()
            ),
            "matrix_exp_calls": sum(
                event.name == "aten::linalg_matrix_exp" for event in prof.events()
            ),
        }
        results.append(record)
        print(json.dumps(record), flush=True)
with open(sys.argv[1], "w") as output:
    json.dump(
        {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "threads": 1,
            "dtype": "float64",
            "results": results,
        },
        output,
        indent=2,
    )
