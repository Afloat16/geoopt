import json
import platform
import sys

import geoopt
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
results = []
manifold = geoopt.CanonicalStiefel()
for batch, n, m in [(8, 64, 4), (4, 256, 8), (1, 1024, 16), (2, 64, 32)]:
    torch.manual_seed(2931)
    x = torch.linalg.qr(torch.randn(batch, n, m, dtype=torch.float64))[0].detach()
    u = manifold.proju(x, 0.1 * torch.randn_like(x)).detach()
    v = manifold.proju(x, torch.randn_like(x)).detach()
    weight = torch.randn_like(x)
    for operation in ["retr", "retr_transp"]:
        for backward in [False, True]:
            x.requires_grad_(backward)
            u.requires_grad_(backward)
            v.requires_grad_(backward)

            def run(
                operation=operation, backward=backward, x=x, u=u, v=v, weight=weight
            ):
                if operation == "retr":
                    output = manifold.retr(x, u)
                    if backward:
                        torch.autograd.grad((output * weight).sum(), (x, u))
                else:
                    point, tangent = manifold.retr_transp(x, u, v)
                    if backward:
                        torch.autograd.grad(
                            ((point + tangent) * weight).sum(), (x, u, v)
                        )

            measurement = Timer(
                stmt="run()", globals={"run": run}, num_threads=1
            ).blocked_autorange(min_run_time=0.3)
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                run()
            solve_shapes = [
                event.input_shapes[0]
                for event in prof.events()
                if event.name == "aten::linalg_solve"
            ]
            result = {
                "operation": operation,
                "backward": backward,
                "batch": batch,
                "rows": n,
                "columns": m,
                "median_ms": measurement.median * 1000,
                "iqr_ms": measurement.iqr * 1000,
                "solve_shapes": solve_shapes,
            }
            results.append(result)
            print(json.dumps(result), flush=True)

with open(sys.argv[1], "w") as output:
    json.dump(
        {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "system": platform.platform(),
            "threads": 1,
            "dtype": "float64",
            "results": results,
        },
        output,
        indent=2,
    )
