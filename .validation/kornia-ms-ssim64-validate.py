"""Run paired source-verified eager public API comparisons."""
import ast
import hashlib
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
candidate="21c8e92cb36341353d910d3922201805bcce4400"
old_source=subprocess.check_output(["git","show","a4c31b02a00e627782b7551bcb5216a2adbf98ab:kornia/losses/ms_ssim.py"],text=True)
new_source=(ROOT/"kornia/losses/ms_ssim.py").read_text()
old_ast=ast.dump(ast.parse(old_source),include_attributes=False)
new_ast=ast.dump(ast.parse(new_source),include_attributes=False)
assert old_ast==new_ast
proof={"measured_head":"a4c31b02a00e627782b7551bcb5216a2adbf98ab","final_head":candidate,"production_ast_sha256":hashlib.sha256(new_ast.encode()).hexdigest(),"identical_ast":True,"scope":"Production source differs only in formatting; numeric test literals now use the observed NumPy values."}
(OUT/"ast-equivalence.json").write_text(json.dumps(proof,indent=2)+"\n")
print("PRODUCTION_AST_EQUIVALENCE "+json.dumps(proof),flush=True)
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
print("FINAL_PUBLIC_API_COMPLETED; timings remain at AST-identical measured head a4c31b02",flush=True)
