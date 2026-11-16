"""Reproduce representable Hausdorff outputs poisoned by half-precision internal weights."""
import json
from pathlib import Path
import subprocess
import sys
import torch
import kornia
from kornia.losses import HausdorffERLoss
torch.set_num_threads(1)
ROOT=Path.cwd()
assert Path(kornia.__file__).resolve() == ROOT/"kornia/__init__.py"
assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()=="b1624c022ceb93373cb073a50bd1da2511108c5e"
print("REPRO_PROVENANCE "+json.dumps({"head":"b1624c022ceb93373cb073a50bd1da2511108c5e","source":str(Path(kornia.__file__).resolve()),"interpreter":sys.executable,"torch":torch.__version__,"scope":"eager CPU original-source reproduction only; no compile, ONNX or product edit"}),flush=True)
H=W=256
side=150
margin=(H-side)//2
positions=torch.arange(H,dtype=torch.float64)
depth=torch.minimum(positions-margin+1,margin+side-positions)
depth=torch.minimum(depth[:,None],depth[None,:]).clamp_min(0)
inside=depth>0
target=torch.zeros(1,1,H,W,dtype=torch.long)
rows=[]
for k in (10,60):
    n=(depth-1).clamp(0,k)
    # Closed form of square weights; no erosion is used by the oracle.
    weight=n*(n+1)*(2*n+1)/6
    expected=(11/16)*weight
    mean=expected.mean()
    expected_grad=torch.stack((-weight,weight))[None]*(7/3)/(2*H*W)
    for dtype in (torch.float64,torch.float32,torch.float16,torch.bfloat16):
        foreground=torch.where(inside,.875,.125).to(dtype)
        pred=torch.stack((1-foreground,foreground))[None].requires_grad_()
        op=HausdorffERLoss(k=k,reduction="none")
        result=op(pred,target)
        loss=HausdorffERLoss(k=k)(pred,target)
        grad,=torch.autograd.grad(loss,pred)
        reference=expected[None,None,None].expand_as(result)
        finite=bool(torch.isfinite(result).all())
        if dtype in (torch.float32,torch.float64):
            torch.testing.assert_close(result.double(),reference,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(loss.double(),mean,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(grad.double(),expected_grad,rtol=1e-6,atol=1e-7)
        if dtype==torch.float16 and k==60:
            assert expected.max()<torch.finfo(dtype).max
            assert not finite and not torch.isfinite(loss)
            assert not torch.isfinite(grad).all()
        row={"k":k,"dtype":str(dtype),"shape":[1,2,H,W],"wrong_square_side":side,"probabilities":[.125,.875],"independent_maximum_loss":expected.max().item(),"independent_mean":mean.item(),"output_finite":finite,"mean_finite":bool(torch.isfinite(loss)),"gradient_finite":bool(torch.isfinite(grad).all()),"nonfinite_output_count":int((~torch.isfinite(result)).sum()),"nonfinite_gradient_count":int((~torch.isfinite(grad)).sum()),"mean":loss.item() if torch.isfinite(loss) else str(loss.item()),"max_absolute_error":(result.double()-reference).abs().max().item() if finite else None}
        rows.append(row)
        print("REPRO_RESULT "+json.dumps(row),flush=True)
out=ROOT.parent/"kornia-hausdorff-reproduction-results"
out.mkdir(exist_ok=True)
(out/"results.json").write_text(json.dumps(rows,indent=2)+"\n")
print("EXPECTED_BASELINE_REGRESSION_REPRODUCED",flush=True)
