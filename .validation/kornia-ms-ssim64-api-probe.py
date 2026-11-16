"""Compare the public MS_SSIMLoss API without compile or ONNX execution."""
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import kornia

ROOT=Path.cwd()
assert Path(kornia.__file__).resolve() == ROOT/"kornia/__init__.py"
head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
assert head == os.environ["EXPECTED_SOURCE_HEAD"]
assert not subprocess.check_output(["git","status","--porcelain","-uno"],text=True).strip()
sys.path.insert(0,str(ROOT/"benchmarks"))
import common
torch.set_num_threads(1)
torch.manual_seed(722)
common.warm_up_cpu()
torch.manual_seed(722)
meta=common.run_metadata(torch.device("cpu"))
meta.update({"full_source_head":head,"python_executable":sys.executable,"kornia_module":"kornia/__init__.py","scope":"public API eager CPU only; compile, ONNX and accelerators unverified","prior_idea":"https://github.com/kornia/kornia/issues/1850#issuecomment-1553684760"})
print("API_PROVENANCE "+json.dumps(meta),flush=True)
out=Path(os.environ["PROBE_OUTPUT"])
out.parent.mkdir(parents=True,exist_ok=True)
rows=[]

def emit(row):
    rows.append(row)
    print("API_RESULT "+json.dumps(row),flush=True)

if os.environ["PROBE_MODE"] == "numeric":
    # Independent full 2-D NumPy correlation, without Kornia or Torch filtering.
    def gauss(s,k):
        g=np.exp(-((np.arange(k)-k//2)**2)/(2*s*s))
        g/=g.sum()
        return np.outer(g,g)
    def filt(a,w):
        k=w.shape[0]
        a=np.pad(a,k//2)
        h,v=a.shape[0]-k+1,a.shape[1]-k+1
        return np.array([[(a[i:i+k,j:j+k]*w).sum() for j in range(v)] for i in range(h)])
    def reference(x,y,sigmas=(.5,1.,2.,4.,8.)):
        k=2*int(2*sigmas[-1])+1
        channels=[]
        l1=[]
        for xc,yc in zip(x,y):
            ms=np.ones(x.shape[1:])
            for s in sigmas:
                w=gauss(s,k)
                mx,my=filt(xc,w),filt(yc,w)
                vx,vy,cxy=filt(xc*xc,w)-mx*mx,filt(yc*yc,w)-my*my,filt(xc*yc,w)-mx*my
                ms*=(2*cxy+.03**2)/(vx+vy+.03**2)
            ms*=(2*mx*my+.01**2)/(mx*mx+my*my+.01**2)
            channels.append(ms)
            l1.append(filt(np.abs(xc-yc),w))
        return 200*(.025*(1-np.mean(channels,axis=0))+.975*np.mean(l1,axis=0))
    x_np=(np.arange(60).reshape(3,4,5)*np.array([1,3,7]).reshape(3,1,1)%11)/10
    expected=reference(x_np,x_np**2)
    independent=torch.from_numpy(expected)[None]
    actual=kornia.losses.MS_SSIMLoss(reduction="none").double()(torch.from_numpy(x_np)[None],torch.from_numpy(x_np**2)[None])
    torch.testing.assert_close(actual,independent,rtol=1e-12,atol=0)
    emit({"kind":"independent_numpy_pin","map":expected.tolist(),"maximum_error":(actual-independent).abs().max().item(),"passed":True})

    snapshots={}
    configs=[
        ("rgb_default",torch.float64,(2,3,12,14),(.5,1.,2.,4.,8.),False,False),
        ("two_channels",torch.float64,(1,2,7,9),(.5,1.,2.,4.,8.),False,False),
        ("four_channels_broadcast",torch.float64,(3,4,12,14),(.5,1.,2.,4.,8.),True,False),
        ("strided_rgb",torch.float64,(1,3,12,14),(.5,1.,2.,4.,8.),False,True),
        ("window17",torch.float64,(1,3,12,14),(.5,1.,2.,4.),False,False),
        ("window15_fallback",torch.float64,(1,3,12,14),(.5,1.,2.,3.9),False,False),
        ("gray_fallback",torch.float64,(1,1,16,16),(.5,1.,2.,4.,8.),False,False),
        ("single_tap_fallback",torch.float64,(1,3,8,8),(.1,),False,False),
        ("float32_fallback",torch.float32,(2,3,24,28),(.5,1.,2.,4.,8.),False,False),
        ("float16_fallback",torch.float16,(1,3,12,14),(.5,1.,2.,4.,8.),False,False),
        ("bfloat16_fallback",torch.bfloat16,(1,3,12,14),(.5,1.,2.,4.,8.),False,False),
    ]
    for name,dtype,shape,sigmas,broadcast,strided in configs:
        for pattern in ("textured","near_identical","constant"):
            torch.manual_seed(722)
            x=torch.rand(shape,dtype=dtype)
            ys=(1,)+shape[1:] if broadcast else shape
            if pattern=="constant":
                x.fill_(.5)
                y=torch.full(ys,.5001,dtype=dtype)
            elif pattern=="near_identical":
                y=((x[:1] if broadcast else x)+.001*torch.randn(ys,dtype=dtype)).clamp(0,1)
            else:
                y=torch.rand(ys,dtype=dtype)
            if strided:
                x,y=x.transpose(-1,-2),y.transpose(-1,-2)
            x.requires_grad_()
            y.requires_grad_()
            for alpha in (.025,1.,0.):
                op=kornia.losses.MS_SSIMLoss(sigmas=sigmas,reduction="none",alpha=alpha).to(dtype)
                result=op(x,y)
                grad=torch.autograd.grad(result.mean(),(x,y))
                key=f"{name}/{pattern}/alpha={alpha}"
                assert torch.isfinite(result).all() and all(torch.isfinite(g).all() for g in grad)
                snapshots[key]={"value":result.detach(),"grad_x":grad[0],"grad_y":grad[1]}
                emit({"kind":"numeric","case":key,"dtype":str(dtype),"mean":result.mean().item(),"minimum":result.min().item(),"maximum":result.max().item(),"grad_x_max":grad[0].abs().max().item(),"finite":True})

    old=kornia.losses.MS_SSIMLoss()
    del old.sigmas
    old=pickle.loads(pickle.dumps(old))
    torch.manual_seed(822)
    x=torch.rand(1,3,12,14,dtype=torch.float64,requires_grad=True)
    y=torch.rand_like(x,requires_grad=True)
    val=old(x,y)
    gx,gy=torch.autograd.grad(val,(x,y))
    snapshots["legacy_pickle_without_sigmas"]={"value":val.detach(),"grad_x":gx,"grad_y":gy}
    emit({"kind":"numeric","case":"legacy_pickle_without_sigmas","value":val.item()})

    torch.save(snapshots,out.with_suffix(".pt"))
else:
    workloads=[
        ("rgb_small",torch.float64,(1,3,16,16),(.5,1.,2.,4.,8.)),
        ("rgb_patch",torch.float64,(1,3,64,64),(.5,1.,2.,4.,8.)),
        ("rgb_batch",torch.float64,(4,3,64,64),(.5,1.,2.,4.,8.)),
        ("rgb_window17",torch.float64,(1,3,64,64),(.5,1.,2.,4.)),
        ("two_channels_window17",torch.float64,(1,2,16,16),(.5,1.,2.,4.)),
        ("gray_fallback",torch.float64,(1,1,16,16),(.5,1.,2.,4.,8.)),
        ("small_window_fallback",torch.float64,(1,3,64,64),(.5,1.)),
        ("single_tap_fallback",torch.float64,(1,1,8,8),(.1,)),
        ("float32_fallback",torch.float32,(1,3,64,64),(.5,1.,2.,4.,8.)),
    ]
    for name,dtype,shape,sigmas in workloads:
        x=torch.rand(shape,dtype=dtype,requires_grad=True)
        y=(x.detach()+.02*torch.randn(shape,dtype=dtype)).clamp(0,1)
        op=kornia.losses.MS_SSIMLoss(sigmas=sigmas).to(dtype)
        for mode in ("forward","forward_backward"):
            def call():
                value=op(x,y)
                return torch.autograd.grad(value,x) if mode=="forward_backward" else value
            median,iqr=common.time_us(call,min_run_time=.6)
            assert np.isfinite(median)
            emit({"kind":"timing","case":name,"dtype":str(dtype),"shape":shape,"sigmas":sigmas,"mode":mode,"median_us":median,"iqr_us":iqr})
common.save_json(out,meta,rows)
print("API_SCOPE_COMPLETE eager-only public API",flush=True)
