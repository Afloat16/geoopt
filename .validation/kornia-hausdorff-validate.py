"""Verify baseline regressions and exact public contracts without compile/export."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch

root = Path.cwd()
base_tree = root.parent / "kornia-base"
validation = root.parent / "validation"
out = root.parent / "kornia-hausdorff-validation-results"
out.mkdir(exist_ok=True)
base = "5cc9c44eddd8dc6a061151d245925f072d673321"
candidate = "ec3b52d4c10f1439e3b72436896fd3ac2bfbac6b"
assert subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip() == candidate
assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()
subprocess.run(["git","worktree","add","--detach",str(base_tree),base],check=True)
assert subprocess.check_output(["git","hash-object","kornia/losses/hausdorff.py"],cwd=base_tree,text=True).strip() == "45eb414075881b7585e81b3d280b37d5901ba5aa"
shutil.copyfile(root/"tests/losses/test_hd.py",base_tree/"tests/losses/test_hd.py")
env = os.environ.copy()
env.update({"CUDA_VISIBLE_DEVICES":"","KORNIA_TEST_OPTIMIZER":"","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1"})
red = subprocess.run([sys.executable,"-m","pytest","tests/losses/test_hd.py","--device=cpu","--dtype=float16,bfloat16","-k","TestHausdorffAccumulation","-q","--junitxml="+str(out/"baseline.xml")],cwd=base_tree,env=env,text=True,capture_output=True)
(out/"baseline.log").write_text(red.stdout+red.stderr)
cases = list(ET.parse(out/"baseline.xml").iter("testcase"))
assert len(cases) == 52
assert not any(case.find("error") is not None for case in cases)
failures = [case.attrib["name"] for case in cases if case.find("failure") is not None]
assert red.returncode == 1
for suffix in [
    "[cpu-float16-none-0.875-HausdorffERLoss-2-256-150-60-2]",
    "[cpu-float16-none-0.875-HausdorffERLoss3D-3-40-30-13-4]",
    "[cpu-float16-mean-0.875-HausdorffERLoss-2-256-150-60-2]",
    "[cpu-float16-mean-0.875-HausdorffERLoss3D-3-40-30-13-4]",
    "[cpu-float16-mean-1.0-HausdorffERLoss-2-256-150-60-2]",
    "[cpu-float16-mean-1.0-HausdorffERLoss3D-3-40-30-13-4]",
]:
    assert any(name.endswith(suffix) for name in failures), suffix
for per_sample in ["False", "True"]:
    assert "test_tensor_exponent_does_not_overflow_before_weighting[cpu-float16-"+per_sample+"]" in failures
assert not any("override" in name or "hook" in name for name in failures)
subprocess.run(["git","restore","--","tests/losses/test_hd.py"],cwd=base_tree,check=True)
print("BASELINE_REGRESSION " + json.dumps({"failed":len(failures),"passed":52-len(failures),"failures":failures}),flush=True)
contract_script=(validation/".validation/kornia-hausdorff-contracts.py").read_text()
contracts={}
for label,tree,head in [("base",base_tree,base),("candidate",root,candidate)]:
    target=out/("contracts-"+label)
    local_env=env|{"EXPECTED_SOURCE_HEAD":head,"CONTRACT_OUTPUT":str(target)}
    result=subprocess.run([sys.executable,"-"],input=contract_script,cwd=tree,env=local_env,text=True,capture_output=True)
    (out/("contracts-"+label+".log")).write_text(result.stdout+result.stderr)
    if result.returncode:
        print(result.stdout[-10000:]+result.stderr[-10000:])
    result.check_returncode()
    contracts[label]=json.loads((target/"contracts.json").read_text())
a,b=contracts["base"]["rows"],contracts["candidate"]["rows"]
assert len(a)==len(b)==288
changed=[]
for old,new in zip(a,b):
    assert old.keys()==new.keys(), (old,new)
    for key in ["ndim","input_dtype","alpha_kind","k","subclass","reduction","output_dtype","direct_dtype","output_shape","direct_shape","calls","gradient_dtype","gradient_finite","error"]:
        assert old.get(key)==new.get(key),(key,old,new)
    if "gradients" in old:
        assert len(old["gradients"])==len(new["gradients"])
        for ga,gb in zip(old["gradients"],new["gradients"]):
            if ga is None:
                assert gb is None
            else:
                assert gb is not None
                assert ga["dtype"]==gb["dtype"] and ga["finite"]==gb["finite"], (old,new)
    if old["input_dtype"] not in ("torch.float16","torch.bfloat16") or old.get("subclass")=="custom_float64":
        assert old==new,(old,new)
    elif old!=new:
        changed.append({"before":old,"after":new})
bench_script=(validation/".validation/kornia-hausdorff-bench.py").read_text()
rounds=[]
for order,(label,tree,head) in enumerate([("base",base_tree,base),("candidate",root,candidate),("candidate",root,candidate),("base",base_tree,base)]):
    filename=out/("bench-"+str(order)+"-"+label+".json")
    result=subprocess.run([sys.executable,"-"],input=bench_script,cwd=tree,env=env|{"EXPECTED_SOURCE_HEAD":head,"BENCH_OUTPUT":str(filename)},text=True,capture_output=True)
    (out/("bench-"+str(order)+".log")).write_text(result.stdout+result.stderr)
    if result.returncode:
        print(result.stdout[-10000:]+result.stderr[-10000:])
    result.check_returncode()
    rounds.append({"label":label,"result":json.loads(filename.read_text())})
cost=[]
for index in range(len(rounds[0]["result"]["rows"])):
    samples={label:[round_["result"]["rows"][index]["median_us"] for round_ in rounds if round_["label"]==label] for label in ("base","candidate")}
    item=rounds[0]["result"]["rows"][index]
    row={"case":item["case"],"dtype":item["dtype"],"shape":item["shape"],"mode":item["mode"],"base_medians_us":samples["base"],"candidate_medians_us":samples["candidate"],"candidate_over_base":statistics.median(samples["candidate"])/statistics.median(samples["base"])}
    cost.append(row)
    print("COST_COMPARISON "+json.dumps(row),flush=True)
summary={"base":base,"candidate":candidate,"torch":torch.__version__,"baseline_failed":len(failures),"baseline_passed":52-len(failures),"contract_rows":288,"all_dtype_shape_error_and_hook_contracts_preserved":True,"full_precision_and_custom_override_rows_bit_exact":True,"half_numeric_rows_changed":len(changed),"cost":cost,"scope":"eager CPU only; no compile, JIT, export, ONNX, CUDA or MPS executed"}
report={"summary":summary,"baseline_failure_ids":failures,"contracts":contracts,"changed_half_rows":changed,"benchmark_rounds":rounds}
raw=(json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
(out/"report.json").write_bytes(raw)
encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
for index in range(0,len(encoded),4000):
    print("HD_AUDIT_CHUNK "+str(index//4000).zfill(4)+" "+encoded[index:index+4000],flush=True)
print("HD_AUDIT_DIGEST "+json.dumps({"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"chunks":(len(encoded)+3999)//4000}),flush=True)
print("HD_VALIDATION_SUMMARY "+json.dumps(summary),flush=True)
