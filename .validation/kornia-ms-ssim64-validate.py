"""Run paired source-verified eager public API comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path.cwd()
BASE=ROOT.parent/"kornia-base"
VALIDATION=ROOT.parent/"validation"
OUT=ROOT.parent/"kornia-ms-ssim64-results"
OUT.mkdir(exist_ok=True)
base="b1624c022ceb93373cb073a50bd1da2511108c5e"
candidate="a4c31b02a00e627782b7551bcb5216a2adbf98ab"
subprocess.run(["git","worktree","add","--detach",str(BASE),base],check=True)
script=(VALIDATION/".validation/kornia-ms-ssim64-api-probe.py").read_text()
python=sys.executable
def probe(tree,head,mode,label):
    env=os.environ.copy()
    env.update({"EXPECTED_SOURCE_HEAD":head,"PROBE_MODE":mode,"PROBE_OUTPUT":str(OUT/f"{label}.json"),"PYTHONDONTWRITEBYTECODE":"1"})
    subprocess.run([python,"-"],input=script,text=True,cwd=tree,env=env,check=True)

probe(BASE,base,"numeric","base-numeric")
probe(ROOT,candidate,"numeric","candidate-numeric")
baseline=torch.load(OUT/"base-numeric.pt",weights_only=True)
patched=torch.load(OUT/"candidate-numeric.pt",weights_only=True)
comparison=[]
assert baseline.keys()==patched.keys()
for case in baseline:
    row={"case":case,"maximum_absolute_difference":{},"bit_exact":{}}
    for key in baseline[case]:
        a,b=baseline[case][key],patched[case][key]
        if "fallback" in case or "legacy_pickle" in case:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        else:
            # Default torch.float64 accuracy is stricter than Kornia's default BaseTester tolerances.
            torch.testing.assert_close(a,b)
        row["maximum_absolute_difference"][key]=(a-b).abs().max().item()
        row["bit_exact"][key]=torch.equal(a,b)
    comparison.append(row)
    print("API_COMPARISON "+json.dumps(row),flush=True)
(OUT/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
# Interleave complete public API timings. Each cell also contains Timer's repeated median/IQR.
order=[("base",BASE,base),("candidate",ROOT,candidate),("candidate",ROOT,candidate),("base",BASE,base)]
for repeat,(label,tree,head) in enumerate(order):
    probe(tree,head,"timing",f"timing-{repeat}-{label}")
print("PAIRED_PUBLIC_API_COMPLETED",flush=True)
