"""Eager-only exploration of the existing separable-Gaussian suggestion.

This changes no Kornia source and does not establish a production-API speedup.
"""
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

import torch
import torch.nn.functional as F
import kornia
from kornia.losses import MS_SSIMLoss

ROOT = Path.cwd()
assert Path(kornia.__file__).resolve() == ROOT / "kornia/__init__.py"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == "b1624c022ceb93373cb073a50bd1da2511108c5e"
sys.path.insert(0, str(ROOT / "benchmarks"))
import common

torch.set_num_threads(1)
torch.manual_seed(4831)
common.warm_up_cpu()
metadata = common.run_metadata(torch.device("cpu"))
metadata.update({
    "python_executable": sys.executable,
    "kornia_module": "kornia/__init__.py",
    "scope": "Eager exploratory subclass, not a production candidate or an accepted performance claim.",
    "prior_idea": "https://github.com/kornia/kornia/issues/1850#issuecomment-1553684760",
})
print("PROVENANCE " + json.dumps(metadata), flush=True)

class ExperimentalSeparableLoss(MS_SSIMLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vectors = torch.stack([self._fspecial_gauss_1d(2*self.pad+1, s) for s in self.sigmas])
        self.register_buffer("_experimental_vectors", vectors, persistent=False)

    def _compute_loss(self, img1, img2, g_masks):
        if img1.is_complex():
            return super()._compute_loss(img1, img2, g_masks)
        if img1.dtype == torch.float64 and self.sigmas:
            vectors = torch.stack([
                self._fspecial_gauss_1d(2*self.pad+1, s, img1.device, torch.float64)
                for s in self.sigmas
            ])
        elif (
            hasattr(self, "_experimental_vectors")
            and self._g_masks.dtype not in (torch.float16, torch.bfloat16)
        ):
            vectors = self._experimental_vectors.to(img1.dtype)
        else:
            # Rounded 2-D half masks and old pickles retain their current arithmetic.
            return super()._compute_loss(img1, img2, g_masks)

        channels = img1.shape[1]
        scales = self.num_scales
        vectors = vectors.repeat(channels, 1)

        def filter_image(image, last_scale=False):
            weights = vectors[scales-1::scales] if last_scale else vectors
            horizontal = F.conv2d(image, weights[:,None,None,:], groups=channels, padding=(0,self.pad))
            return F.conv2d(horizontal, weights[:,None,:,None], groups=weights.shape[0], padding=(self.pad,0))

        mux = filter_image(img1)
        muy = filter_image(img2)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        vx = filter_image(img1 * img1) - mux2
        vy = filter_image(img2 * img2) - muy2
        cov = filter_image(img1 * img2) - muxy
        luminance = (2*muxy + self.C1)/(mux2 + muy2 + self.C1)
        contrast = (2*cov + self.C2)/(vx + vy + self.C2)
        ms = 1 - (luminance[:,scales-1::scales] * contrast.unflatten(1,[channels,scales]).prod(2)).mean(1)
        l1 = filter_image((img1-img2).abs(), last_scale=True).mean(1)
        loss = self.compensation * (self.alpha*ms + (1-self.alpha)*l1/self.DR)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise NotImplementedError(self.reduction)

results = []
def emit(row):
    results.append(row)
    print("PROBE_RESULT " + json.dumps(row), flush=True)

# Independent NumPy-generated pin already documented by upstream PR #6610.
x = ((torch.arange(64,dtype=torch.float64)*0.37+0.11)%1).reshape(1,1,8,8)
y = ((torch.arange(64,dtype=torch.float64)*0.61+0.05)%1).reshape(1,1,8,8)
expected = 54.83400477404822
pin = ExperimentalSeparableLoss(sigmas=(0.3,0.7,1.5)).double()(x,y).item()
assert math.isclose(pin, expected, rel_tol=1e-12, abs_tol=0), (pin,expected)
emit({"kind":"independent_pin","expected":expected,"actual":pin,"passed":True})

x = torch.rand(1,1,5,5,dtype=torch.float64,requires_grad=True)
y = torch.rand_like(x,requires_grad=True)
gradcheck = torch.autograd.gradcheck(ExperimentalSeparableLoss(sigmas=(0.5,1.0,2.0)).double(),(x,y))
emit({"kind":"finite_difference_gradcheck","passed":gradcheck})
assert gradcheck

for dtype in (torch.float32,torch.float64):
    for pattern in ("textured","near_identical","constant"):
        for broadcast in (False,True):
            shape=(2,3,24,28)
            x=torch.rand(shape,dtype=dtype)
            if pattern=="constant":
                x.fill_(0.5)
                y=torch.full((1,3,24,28) if broadcast else shape,0.5001,dtype=dtype)
            elif pattern=="near_identical":
                y=(x[:1] if broadcast else x)+0.001*torch.randn((1,3,24,28) if broadcast else shape,dtype=dtype)
                y=y.clamp(0,1)
            else:
                y=torch.rand((1,3,24,28) if broadcast else shape,dtype=dtype)
            x.requires_grad_()
            a=MS_SSIMLoss().to(dtype)
            b=ExperimentalSeparableLoss().to(dtype)
            av=a(x,y); bv=b(x,y)
            ag,=torch.autograd.grad(av,x); bg,=torch.autograd.grad(bv,x)
            reference=MS_SSIMLoss().double()(x.detach().double(),y.double())
            emit({
                "kind":"value_gradient_comparison","dtype":str(dtype),"pattern":pattern,"broadcast":broadcast,
                "base":av.item(),"experimental":bv.item(),"value_absolute_difference":(av-bv).abs().item(),
                "base_float64_reference_error":abs(av.item()-reference.item()),
                "experimental_float64_reference_error":abs(bv.item()-reference.item()),
                "gradient_max_absolute_difference":(ag-bg).abs().max().item(),
                "gradient_max_absolute_scale":ag.abs().max().item(),
                "finite":bool(torch.isfinite(bv).all() and torch.isfinite(bg).all()),
            })

for dtype in (torch.float16,torch.bfloat16):
    x=torch.rand(1,3,16,16,dtype=dtype)
    y=torch.rand_like(x)
    av=MS_SSIMLoss().to(dtype)(x,y)
    bv=ExperimentalSeparableLoss().to(dtype)(x,y)
    assert torch.equal(av,bv)
    emit({"kind":"half_mask_fallback","dtype":str(dtype),"exact_match":True})

old=ExperimentalSeparableLoss()
del old._buffers["_experimental_vectors"]
old=pickle.loads(pickle.dumps(old))
x=torch.rand(1,3,16,16);y=torch.rand_like(x)
assert torch.equal(old(x,y),MS_SSIMLoss()(x,y))
emit({"kind":"pickle_without_vector_cache","exact_match":True})

previous=torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
try:
    base_cast=MS_SSIMLoss().float()
    experimental_cast=ExperimentalSeparableLoss().float()
finally:
    torch.set_default_dtype(previous)
av=base_cast(x,y);bv=experimental_cast(x,y)
emit({"kind":"float64_constructed_float32_module","base":av.item(),"experimental":bv.item(),"absolute_difference":(av-bv).abs().item()})

workloads=[
 ("gray_patch",(1,1,16,16),(0.5,1.0,2.0,4.0,8.0)),
 ("rgb_patch",(1,3,64,64),(0.5,1.0,2.0,4.0,8.0)),
 ("rgb_batch",(4,3,64,64),(0.5,1.0,2.0,4.0,8.0)),
 ("rgb_image",(1,3,192,256),(0.5,1.0,2.0,4.0,8.0)),
 ("small_window",(1,3,64,64),(0.5,1.0)),
 ("single_tap",(1,1,8,8),(0.1,)),
]
for name,shape,sigmas in workloads:
    x=torch.rand(shape,requires_grad=True)
    y=(x.detach()+0.02*torch.randn(shape)).clamp(0,1)
    base=MS_SSIMLoss(sigmas=sigmas)
    experimental=ExperimentalSeparableLoss(sigmas=sigmas)
    for mode in ("forward","forward_backward"):
        def operation(module):
            value=module(x,y)
            return torch.autograd.grad(value,x) if mode=="forward_backward" else value
        order=[("base",base),("experimental",experimental),("experimental",experimental),("base",base)]
        measurements=[]
        for label,module in order:
            median,iqr=common.time_us(lambda:operation(module),min_run_time=0.35)
            assert math.isfinite(median)
            measurements.append({"implementation":label,"median_us":median,"iqr_us":iqr})
        emit({"kind":"exploratory_timing","workload":name,"shape":shape,"sigmas":sigmas,"mode":mode,"measurements":measurements})

OUT=ROOT.parent/"kornia-ms-ssim-eager-results"
OUT.mkdir(exist_ok=True)
common.save_json(OUT/"probe.json",metadata,results)
print("SCOPE_COMPLETE eager values, numerical gradient check, negative controls, and exploratory timings only",flush=True)
