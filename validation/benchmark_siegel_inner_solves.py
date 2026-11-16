import json
import platform
import sys

import geoopt
import torch
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
results = []
for model in ["bounded", "upper"]:
    manifold = geoopt.BoundedDomain() if model == "bounded" else geoopt.UpperHalf()
    for batch, size in [(16, 32), (4, 128), (1, 256)]:
        torch.manual_seed(1752)
        raw = torch.randn(batch, size, size, dtype=torch.complex128)
        symmetric = (raw + raw.transpose(-1, -2)) / 2
        if model == "bounded":
            base = symmetric / (2 * torch.linalg.matrix_norm(symmetric, keepdim=True))
        else:
            y = symmetric.imag @ symmetric.imag.transpose(-1, -2) / size + torch.eye(
                size
            )
            base = torch.complex(symmetric.real, y)
        raw_u = torch.randn_like(raw)
        raw_v = torch.randn_like(raw)
        u_base = (raw_u + raw_u.transpose(-1, -2)) / 2
        v_base = (raw_v + raw_v.transpose(-1, -2)) / 2
        for norm in [False, True]:
            for backward in [False, True]:
                z = base.detach().requires_grad_(backward)
                u = u_base.detach().requires_grad_(backward)
                v = None if norm else v_base.detach().requires_grad_(backward)

                def run(manifold=manifold, z=z, u=u, v=v, backward=backward):
                    result = manifold.inner(z, u, v)
                    if backward:
                        torch.autograd.grad(
                            result.real.sum(), (z, u) if v is None else (z, u, v)
                        )

                timing = Timer(
                    stmt="run()", globals={"run": run}, num_threads=1
                ).blocked_autorange(min_run_time=0.3)
                row = {
                    "model": model,
                    "batch": batch,
                    "size": size,
                    "norm": norm,
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
