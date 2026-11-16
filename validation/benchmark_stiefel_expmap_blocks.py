import json
import platform
import sys

import geoopt
import torch
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
manifold = geoopt.EuclideanStiefel()
results = []
for batch, rows, columns in [(16, 128, 8), (4, 512, 16), (1, 2048, 16), (4, 32, 16)]:
    torch.manual_seed(1741)
    x_base = torch.linalg.qr(torch.randn(batch, rows, columns, dtype=torch.float64)).Q
    u_base = manifold.proju(x_base, 0.03 * torch.randn_like(x_base))
    weight = torch.randn_like(x_base)
    for backward in [False, True]:
        x = x_base.detach().requires_grad_(backward)
        u = u_base.detach().requires_grad_(backward)

        def run(x=x, u=u, weight=weight, backward=backward):
            result = manifold.expmap(x, u)
            if backward:
                torch.autograd.grad((result * weight).sum(), (x, u))

        timing = Timer(
            stmt="run()", globals={"run": run}, num_threads=1
        ).blocked_autorange(min_run_time=0.4)
        item = {
            "batch": batch,
            "rows": rows,
            "columns": columns,
            "backward": backward,
            "median_ms": timing.median * 1000,
            "iqr_ms": timing.iqr * 1000,
        }
        results.append(item)
        print(json.dumps(item), flush=True)
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
