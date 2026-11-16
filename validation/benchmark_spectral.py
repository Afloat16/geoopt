import json
import platform
import sys

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.benchmark import Timer

from geoopt import linalg

torch.manual_seed(0)
torch.set_num_threads(1)
functions = {"log": linalg.sym_logm, "paired_roots": linalg.sym_inv_sqrtm2}
results = []
for name, function in functions.items():
    for batch, size in ((8, 64), (4, 256), (1, 512)):
        raw = torch.randn(batch, size, size, dtype=torch.float64)
        matrix = raw @ raw.transpose(-1, -2) + torch.eye(size, dtype=raw.dtype)
        with torch.no_grad():
            for _ in range(10):
                function(matrix)
            timing = Timer(stmt="function(matrix)", globals={"function": function, "matrix": matrix}, num_threads=1).blocked_autorange(min_run_time=0.5)
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as trace:
                function(matrix)
        events = trace.key_averages()
        diagonal_calls = sum(event.count for event in events if event.key == "aten::diag_embed")
        matmul_calls = sum(event.count for event in events if event.key in ("aten::mm", "aten::bmm"))
        results.append({"name": name, "batch": batch, "size": size, "median_ms": timing.median * 1000, "diag_embed_calls": diagonal_calls, "dense_matmul_calls": matmul_calls})
data = {"torch": torch.__version__, "platform": platform.platform(), "threads": 1, "results": results}
with open(sys.argv[1], "w") as stream:
    json.dump(data, stream, indent=2)
print(json.dumps(data))
