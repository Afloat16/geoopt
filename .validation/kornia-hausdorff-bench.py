"""Measure ordinary eager CPU call cost at exact source revisions."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import kornia
from benchmarks.common import time_us, warm_up_cpu

root = Path.cwd()
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert head == os.environ["EXPECTED_SOURCE_HEAD"]
assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
assert Path(kornia.__file__).resolve() == root / "kornia/__init__.py"
torch.set_num_threads(1)
torch.manual_seed(0)
warm_up_cpu()
rows = []
for name, cls, batch, spatial, margin in [
    ("2d-default-k10", kornia.losses.HausdorffERLoss, 2, (64, 64), 8),
    ("3d-default-k10", kornia.losses.HausdorffERLoss3D, 1, (24, 24, 24), 1),
]:
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        foreground = torch.zeros((batch, *spatial), dtype=dtype)
        foreground[(slice(None), *[slice(margin, n-margin) for n in spatial])] = .875
        pred = torch.stack((1-foreground, foreground), 1)
        target = torch.zeros((batch, 1, *spatial), dtype=torch.long)
        op = cls()
        assert torch.isfinite(op(pred, target))
        train_pred = pred.clone().requires_grad_()
        def forward():
            return op(pred, target)
        def training():
            return torch.autograd.grad(op(train_pred, target), train_pred)[0]
        assert torch.isfinite(training()).all()
        for mode, fn in [("forward", forward), ("forward_backward", training)]:
            median, iqr = time_us(fn, min_run_time=1.0, num_threads=1)
            assert math.isfinite(median) and median > 0 and math.isfinite(iqr)
            row = {"case": name, "shape": list(pred.shape), "dtype": str(dtype), "mode": mode, "median_us": median, "iqr_us": iqr}
            rows.append(row)
            print("BENCH_RESULT " + json.dumps(row), flush=True)
result = {"head": head, "source": str(Path(kornia.__file__).resolve()), "interpreter": sys.executable, "torch": torch.__version__, "threads": 1, "min_run_seconds": 1.0, "rows": rows}
Path(os.environ["BENCH_OUTPUT"]).write_text(json.dumps(result, indent=2)+"\n")
