"""Capture the unchanged Hausdorff dtype, alpha-promotion and subclass contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import kornia

torch.set_num_threads(1)
root = Path.cwd()
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
expected_head = os.environ.get("EXPECTED_SOURCE_HEAD", "b1624c022ceb93373cb073a50bd1da2511108c5e")
assert head == expected_head
assert Path(kornia.__file__).resolve() == root / "kornia/__init__.py"
assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
provenance = {"head": head, "source": str(Path(kornia.__file__).resolve()), "interpreter": sys.executable, "torch": torch.__version__, "scope": "eager CPU dtype/alpha/subclass public contracts; no compile, tracing, export or ONNX"}
print("CONTRACT_PROVENANCE " + json.dumps(provenance), flush=True)
rows = []
for ndim, cls in [(2, kornia.losses.HausdorffERLoss), (3, kornia.losses.HausdorffERLoss3D)]:
    spatial = (8,) * ndim
    target = torch.zeros(2, 1, *spatial, dtype=torch.long)
    for dtype in [torch.bool, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        foreground = torch.full((2, *spatial), .875 if dtype.is_floating_point else 1, dtype=dtype)
        if dtype == torch.bool:
            pred0 = torch.stack((~foreground, foreground), 1)
        else:
            pred0 = torch.stack((1 - foreground, foreground), 1)
        for alpha_kind in ["float", "int", "numpy64", "scalar_f16", "scalar_f32", "scalar_f64", "batch_f16", "batch_bf16", "batch_f32", "batch_f64"]:
            for k in [0, 3]:
                if alpha_kind == "float":
                    alpha = 2.0
                elif alpha_kind == "int":
                    alpha = 2
                elif alpha_kind == "numpy64":
                    alpha = np.float64(2)
                else:
                    adtype = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}[alpha_kind.split("_")[1]]
                    if alpha_kind.startswith("batch"):
                        alpha = torch.tensor([2., -1.], dtype=adtype).view(2, 1, *([1] * ndim)).requires_grad_()
                    else:
                        alpha = torch.tensor(2., dtype=adtype, requires_grad=True)
                pred = pred0.clone().requires_grad_(dtype.is_floating_point)
                row = {"ndim": ndim, "input_dtype": str(dtype), "alpha_kind": alpha_kind, "k": k}
                try:
                    op = cls(k=k, alpha=alpha)
                    value = op(pred, target)
                    direct = op.perform_erosion(pred[:, :1], (target == 0).long())
                    row.update({"output_dtype": str(value.dtype), "direct_dtype": str(direct.dtype), "output_shape": list(value.shape), "direct_shape": list(direct.shape), "value": value.item()})
                    inputs = ([pred] if pred.requires_grad else []) + ([alpha] if isinstance(alpha, torch.Tensor) and alpha.requires_grad else [])
                    if value.requires_grad and inputs:
                        grads = torch.autograd.grad(value, inputs, allow_unused=True)
                        row["gradients"] = [None if g is None else {"dtype": str(g.dtype), "sum": g.double().sum().item(), "absmax": g.double().abs().max().item(), "finite": bool(torch.isfinite(g).all())} for g in grads]
                except Exception as exc:
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                rows.append(row)
                print("CONTRACT_RESULT " + json.dumps(row), flush=True)
    for kind in ["passthrough", "custom_float64"]:
        class Override(cls):
            def perform_erosion(self, pred, target):
                self.seen.append((str(pred.dtype), str(target.dtype), list(pred.shape)))
                if kind == "passthrough":
                    return super().perform_erosion(pred, target)
                return (pred - target).double().square()

        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for reduction in ["none", "mean", "sum"]:
                pred = torch.tensor([.125, .875], dtype=dtype).view(1, 2, *([1] * ndim)).expand(2, 2, *spatial).clone().requires_grad_()
                op = Override(k=3, reduction=reduction)
                op.seen = []
                value = op(pred, target)
                grad, = torch.autograd.grad(value.sum(), pred)
                row = {"ndim": ndim, "subclass": kind, "input_dtype": str(dtype), "reduction": reduction, "calls": op.seen, "output_dtype": str(value.dtype), "output_shape": list(value.shape), "output_mean": value.double().mean().item(), "gradient_dtype": str(grad.dtype), "gradient_finite": bool(torch.isfinite(grad).all())}
                rows.append(row)
                print("CONTRACT_RESULT " + json.dumps(row), flush=True)
out = Path(os.environ["CONTRACT_OUTPUT"]) if "CONTRACT_OUTPUT" in os.environ else root.parent / "kornia-hausdorff-contract-results"
out.mkdir(exist_ok=True)
(out / "contracts.json").write_text(json.dumps({"provenance": provenance, "rows": rows}, indent=2) + "\n")
print("CONTRACT_COMPLETE " + str(len(rows)), flush=True)
