import json
import sys

import geoopt
import torch
import torch.utils.benchmark as benchmark


torch.manual_seed(1914)
torch.set_num_threads(1)
manifold = geoopt.CanonicalStiefel()
results = []
for shape in [(32, 32), (256, 16), (1024, 16), (4, 256, 16)]:
    x = torch.linalg.qr(torch.randn(*shape, dtype=torch.float64)).Q
    u = torch.randn_like(x)
    for backward in [False, True]:
        a = x.detach().requires_grad_(backward)
        b = u.detach().requires_grad_(backward)
        weight = torch.randn_like(a)

        def run():
            result = manifold.egrad2rgrad(a, b)
            if backward:
                return torch.autograd.grad((result * weight).sum(), (a, b))
            return result

        for _ in range(3):
            run()
        timing = benchmark.Timer(stmt="run()", globals={"run": run}, num_threads=1).blocked_autorange(min_run_time=0.5)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            run()
        matrix_product_bytes = max(
            event.cpu_memory_usage for event in prof.events() if event.name in ("aten::mm", "aten::bmm")
        )
        results.append({"shape": shape, "backward": backward, "median_ms": timing.median * 1000, "largest_product_bytes": matrix_product_bytes})
payload = {"torch": torch.__version__, "results": results}
with open(sys.argv[1], "w") as output:
    json.dump(payload, output, indent=2)
print(json.dumps(payload))
