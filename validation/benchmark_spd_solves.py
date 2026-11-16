import json
import platform
import sys

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

import geoopt

torch.manual_seed(0)
torch.set_num_threads(1)
manifold = geoopt.SymmetricPositiveDefinite()
results = []
for name in ("self_inner", "cross_inner", "retraction"):
    for backward in (False, True):
        for batch, size in ((8, 64), (2, 256), (1, 512)):
            raw = torch.randn(batch, size, size, dtype=torch.float64)
            x = (raw @ raw.transpose(-1, -2) + torch.eye(size, dtype=raw.dtype)).requires_grad_()
            u0 = torch.randn_like(x)
            u = ((u0 + u0.transpose(-1, -2)) * 0.05).requires_grad_()
            v0 = torch.randn_like(x)
            v = ((v0 + v0.transpose(-1, -2)) * 0.05).requires_grad_()
            inputs = (x, u, v) if name == "cross_inner" else (x, u)

            def run():
                with torch.set_grad_enabled(backward):
                    if name == "self_inner":
                        value = manifold.inner(x, u)
                    elif name == "cross_inner":
                        value = manifold.inner(x, u, v)
                    else:
                        value = manifold.retr(x, u)
                    if backward:
                        return torch.autograd.grad(value.sum(), inputs)
                    return value

            for _ in range(3):
                run()
            timing = Timer(stmt="run()", globals={"run": run}, num_threads=1).blocked_autorange(min_run_time=0.35)
            with profile(activities=[ProfilerActivity.CPU]) as trace:
                run()
            events = trace.key_averages()
            inversion_calls = sum(event.count for event in events if event.key == "aten::linalg_inv_ex")
            matmul_calls = sum(event.count for event in events if event.key in ("aten::mm", "aten::bmm"))
            results.append({"name": name, "backward": backward, "batch": batch, "size": size, "median_ms": timing.median * 1000, "inversion_calls": inversion_calls, "dense_matmul_calls": matmul_calls})
data = {"torch": torch.__version__, "platform": platform.platform(), "dtype": "float64", "threads": 1, "results": results}
with open(sys.argv[1], "w") as stream:
    json.dump(data, stream, indent=2)
print(json.dumps(data))
