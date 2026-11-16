"""Verify the two independent candidates with CPU unit tests and public-API measurements."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

BASE = "cd2349f67432f0f5b43d2388634aa231613e3cb0"
PATCH_HEAD = "2cf647172905c7260042ed9fa160547b64e579e1"
MAP_HEAD = "1f03856ec4a65a1bcaf22f3a46372c5825ebe6c6"
root = Path.cwd().resolve()
harness = Path(__file__).resolve().parent
output = root.parent / "kornia-validation-results"
output.mkdir(exist_ok=True)
base = root.parent / "kornia-base"
map_tree = root.parent / "kornia-map"
python = str(Path(sys.executable).absolute())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


assert git("rev-parse", "HEAD") == PATCH_HEAD
subprocess.run(["git", "fetch", "--no-tags", "--depth=2", "origin", MAP_HEAD], cwd=root, check=True)
subprocess.run(["git", "worktree", "add", "--detach", str(base), BASE], cwd=root, check=True)
subprocess.run(["git", "worktree", "add", "--detach", str(map_tree), MAP_HEAD], cwd=root, check=True)
for tree in (base, root, map_tree):
    code = (
        "from pathlib import Path; import kornia,sys,torch; "
        "print(kornia.__file__,sys.executable,torch.__version__); "
        "assert Path(kornia.__file__).resolve().parent == Path.cwd()/'kornia'"
    )
    subprocess.run([python, "-c", code], cwd=tree, check=True)


def benchmark(tree, script, label, only=""):
    destination = output / (label + ".json")
    command = [python, "-", "--json", str(destination)]
    if only:
        command.extend(["--only", only])
    with (output / (label + ".log")).open("w") as log:
        subprocess.run(command, input=(harness / script).read_text(), cwd=tree, text=True,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    value = json.loads(destination.read_text())
    for row in value["results"]:
        for key, number in row.items():
            if key.endswith("median_us"):
                assert math.isfinite(number) and number > 0, (label, row)
    return value


baseline = benchmark(base, "patch-count-benchmark.py", "patch-base")
candidate = benchmark(root, "patch-count-benchmark.py", "patch-candidate")
patch_results = []
for before, after in zip(baseline["results"], candidate["results"], strict=True):
    assert before["name"] == after["name"]
    for field in ("output_sha256", "gradient_sha256"):
        assert before[field] == after[field], (before["name"], field)
    patch_results.append({
        "name": before["name"],
        "forward_speedup": before["forward_median_us"] / after["forward_median_us"],
        "forward_backward_speedup": before["forward_backward_median_us"] / after["forward_backward_median_us"],
        "base": before,
        "candidate": after,
    })

gray = {"base": [], "candidate": []}
for repetition, order in enumerate((("candidate", "base"), ("base", "candidate"), ("candidate", "base"))):
    for label in order:
        tree = base if label == "base" else root
        sample = benchmark(tree, "patch-count-benchmark.py", f"gray-{repetition}-{label}", "gray")
        gray[label].append(sample["results"][0])
gray_comparison = {}
for mode in ("forward", "forward_backward"):
    key = mode + "_median_us"
    a = [row[key] for row in gray["base"]]
    b = [row[key] for row in gray["candidate"]]
    gray_comparison[mode] = {"base_samples_us": a, "candidate_samples_us": b,
                             "median_speedup": statistics.median(a) / statistics.median(b)}

map_baseline = benchmark(base, "map-count-benchmark.py", "map-base")
map_candidate = benchmark(map_tree, "map-count-benchmark.py", "map-candidate")
map_results = []
for before, after in zip(map_baseline["results"], map_candidate["results"], strict=True):
    assert before["name"] == after["name"]
    assert after["actual"] == after["oracle_in_output_dtype"], after
    map_results.append({"name": before["name"], "speedup": before["median_us"] / after["median_us"],
                        "base": before, "candidate": after})


def tests(tree, label, paths, extra=(), expected_exit=0, required_summary=""):
    command = [python, "-m", "pytest", *paths, "--device=cpu", "-q", "--tb=short", "--color=no", *extra]
    result = subprocess.run(command, cwd=tree, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / (label + ".log")).write_text(result.stdout)
    print(label, "\n", result.stdout[-8000:], flush=True)
    assert result.returncode == expected_exit, (label, result.returncode)
    if required_summary:
        assert required_summary in result.stdout, (label, required_summary)
    return {"label": label, "exit_code": result.returncode, "summary": result.stdout.splitlines()[-1]}


checks = []
patch_tests = ["tests/contrib/test_combine_tensor_patch.py", "tests/contrib/test_extract_tensor_patch.py"]
shutil.copy2(root / patch_tests[0], base / patch_tests[0])
checks.append(tests(base, "patch-baseline-tests", patch_tests, ("--dtype=float32,float64",)))
checks.append(tests(root, "patch-candidate-tests", patch_tests, ("--dtype=float32,float64",)))
shutil.copy2(map_tree / "tests/metrics/test_map.py", base / "tests/metrics/test_map.py")
checks.append(tests(base, "map-baseline-regressions", ["tests/metrics/test_map.py"],
                    ("-k", "TestMAPAccumulationPrecision"), expected_exit=1, required_summary="4 failed, 2 passed"))
checks.append(tests(map_tree, "map-candidate-tests", ["tests/metrics/test_map.py", "tests/metrics/test_mean_iou.py"],
                    ("--dtype=float16,bfloat16,float32,float64",)))

summary = {
    "base": BASE, "patch_head": PATCH_HEAD, "map_head": MAP_HEAD,
    "environment": candidate["metadata"], "checks": checks,
    "patch_measurements": patch_results, "gray_interleaved_controls": gray_comparison,
    "map_measurements": map_results,
    "scope": "CPU eager unit tests, TorchScript tests present in the selected files, and public API benchmarks.",
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
