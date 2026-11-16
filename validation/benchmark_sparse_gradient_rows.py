import json
import platform
import sys

import geoopt
import torch
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
results = []
for kind in ["sgd", "adam"]:
    for entries in [128, 4096]:
        for coalesced in [False, True]:
            torch.manual_seed(1852)
            point = torch.nn.Parameter(torch.randn(50000, 64, dtype=torch.float64))
            rows = torch.randint(50000, (entries,)).repeat(2)
            values = torch.randn(rows.numel(), 64, dtype=point.dtype)
            gradient = torch.sparse_coo_tensor(rows.unsqueeze(0), values, point.shape)
            if coalesced:
                gradient = gradient.coalesce()
            point.grad = gradient
            optimizer_class = (
                geoopt.optim.SparseRiemannianSGD
                if kind == "sgd"
                else geoopt.optim.SparseRiemannianAdam
            )
            optimizer = optimizer_class([point], lr=0.001)
            for _ in range(3):
                optimizer.step()
            timing = Timer(
                stmt="optimizer.step()", globals={"optimizer": optimizer}, num_threads=1
            ).blocked_autorange(min_run_time=0.5)
            row = {
                "optimizer": kind,
                "entries": entries,
                "coalesced": coalesced,
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
            "dtype": "float64",
            "results": results,
        },
        output,
        indent=2,
    )
