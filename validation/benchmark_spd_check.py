import json
import sys

import torch
from torch.utils.benchmark import Timer

import geoopt


torch.set_num_threads(1)
manifold = geoopt.SymmetricPositiveDefinite()
results = []
for shape in [(16, 16), (128, 128), (512, 512), (1024, 1024), (8, 128, 128)]:
    for requires_grad in [False, True]:
        generator = torch.Generator().manual_seed(912)
        a = torch.randn(shape, generator=generator, dtype=torch.float64)
        point = a @ a.transpose(-1, -2) + torch.eye(shape[-1], dtype=torch.float64)
        point.requires_grad_(requires_grad)
        assert manifold.check_point_on_manifold(point)
        timing = Timer(
            stmt="manifold.check_point_on_manifold(point)",
            globals={"manifold": manifold, "point": point},
            num_threads=1,
        ).blocked_autorange(min_run_time=0.25)
        record = {
            "shape": shape,
            "requires_grad": requires_grad,
            "stage": sys.argv[1],
            "median_ms": timing.median * 1000,
            "torch": torch.__version__,
        }
        results.append(record)
        print(json.dumps(record), flush=True)

with open(sys.argv[1], "w") as stream:
    json.dump({"results": results}, stream, indent=2)
