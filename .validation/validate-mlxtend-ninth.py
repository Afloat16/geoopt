import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

payload_path, candidate_name = sys.argv[1:3]
payload = next(x for x in json.loads(Path(payload_path).read_text()) if x["name"] == candidate_name)
project = Path("project").resolve()
results = project / ".validation-results"
results.mkdir(exist_ok=True)
os.chdir(project)

def run(name, args, required=True):
    process = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (results / (name + ".log")).write_text(process.stdout)
    print(f"=== {name}: exit {process.returncode} ===", flush=True)
    print(process.stdout[-16000:], flush=True)
    if required and process.returncode:
        raise RuntimeError(name + " failed")
    return process.returncode

def write_files(paths):
    for path in paths:
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload["files"][path])

def report(name):
    path = results / (name + ".xml")
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failures = {}
    for case in cases:
        error = case.find("failure")
        if error is None:
            error = case.find("error")
        if error is not None:
            failures[case.get("classname", "") + "::" + case.get("name", "")] = error.get("message", "")
    return {"total": len(cases), "failed": len(failures), "skipped": sum(x.find("skipped") is not None for x in cases), "failures": failures}

uv = ["uv", "run", "--frozen", "--group", "dev", "python"]
run("versions", uv + ["-c", "import sys,numpy,scipy,sklearn,pandas; print(sys.version); print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'pandas',pandas.__version__)"])
if payload.get("benchmark"):
    run("baseline-benchmark", uv + ["-c", payload["benchmark"]])
write_files(payload.get("fixtures", []))
write_files(payload["tests"])
baseline_code = run("baseline-focused", uv + ["-m", "pytest", "-q", *payload["tests"], "--junitxml=" + str(results / "baseline-focused.xml")], required=False)
baseline_focused = report("baseline-focused")
if payload.get("expect_baseline_failure", True):
    assert baseline_code == 1 and baseline_focused["failed"] > 0, baseline_focused
else:
    assert baseline_code == 0 and baseline_focused["failed"] == 0, baseline_focused
run("baseline-full", uv + ["-m", "pytest", "mlxtend", *payload.get("full_extra", []), "-q", "--continue-on-collection-errors", "--junitxml=" + str(results / "baseline-full.xml")], required=False)
baseline_full = report("baseline-full")
write_files(payload["files"])
changed_python = [p for p in payload["files"] if p.endswith(".py")]
run("isort-format", uv + ["-m", "isort", "-p", "mlxtend", "--line-length", "88", "--multi-line", "3", "--profile", "black", *changed_python])
run("black-format", uv + ["-m", "black", *changed_python])
for path in payload["files"]:
    payload["files"][path] = (project / path).read_text()
(results / "formatted-payload.json").write_text(json.dumps(payload, indent=2))
print("FORMATTED_PAYLOAD=" + json.dumps(payload), flush=True)
if payload.get("benchmark"):
    run("candidate-benchmark", uv + ["-c", payload["benchmark"]])
    def benchmark_result(name):
        line = next(line for line in (results / (name + ".log")).read_text().splitlines() if line.startswith("BENCHMARK_JSON="))
        return json.loads(line.split("=", 1)[1])
    before = benchmark_result("baseline-benchmark")
    after = benchmark_result("candidate-benchmark")
    if before["kind"] == "memory":
        assert after["peak_bytes"] < before["peak_bytes"] * 0.2, (before, after)
    else:
        for old, new in zip(before["results"], after["results"]):
            assert old["digest"] == new["digest"], (old, new)
            assert new["median_seconds"] < old["median_seconds"] * 0.95, (old, new)
    if before["kind"] != "memory" and "peak_bytes" in before:
        assert after["peak_bytes"] < before["peak_bytes"] * 0.3, (before, after)
    performance = {"before": before, "after": after}
    (results / "performance.json").write_text(json.dumps(performance, indent=2))
    print("PERFORMANCE_RESULT=" + json.dumps(performance), flush=True)
run("candidate-focused", uv + ["-m", "coverage", "run", "--source=mlxtend", "--branch", "-m", "pytest", "-q", *payload["adjacent"], "--junitxml=" + str(results / "candidate-focused.xml")])
run("coverage", uv + ["-m", "coverage", "report", "--include=" + ",".join(payload["production"])])
run("coverage-xml", uv + ["-m", "coverage", "xml", "-o", str(results / "coverage.xml")])
run("candidate-full", uv + ["-m", "pytest", "mlxtend", *payload.get("full_extra", []), "-q", "--continue-on-collection-errors", "--junitxml=" + str(results / "candidate-full.xml")], required=False)
candidate_full = report("candidate-full")
introduced = set(candidate_full["failures"]) - set(baseline_full["failures"])
assert not introduced, sorted(introduced)
assert candidate_full["failed"] == 0, candidate_full
run("isort-check", uv + ["-m", "isort", "-p", "mlxtend", "--check", "--diff", "--line-length", "88", "--multi-line", "3", "--profile", "black", *changed_python])
run("black-check", uv + ["-m", "black", "--check", "--diff", *changed_python])
run("flake8", uv + ["-m", "flake8", "--config=.flake8", *changed_python])
run("repository-isort-check", uv + ["-m", "isort", "-p", "mlxtend", "--check", "--diff", "--line-length", "88", "--multi-line", "3", "--py", "39", "--profile", "black", *map(str, sorted(Path("mlxtend").glob("*")))])
run("repository-black-check", uv + ["-m", "black", "--check", "--diff", *map(str, sorted(Path("mlxtend").glob("*")))])
run("repository-flake8-report", uv + ["-m", "flake8", ".", "--config=.flake8", "--count", "--exit-zero", "--statistics"])
run("diff-check", ["git", "diff", "--check"])
run("build", ["uv", "build"])
run("twine", uv + ["-m", "twine", "check", "--strict", *map(str, sorted(Path("dist").glob("*.whl")) + sorted(Path("dist").glob("*.tar.gz")))])
summary = {"name": payload["name"], "base": payload["base"], "issue": payload["issue"],
           "baseline_focused": baseline_focused, "baseline_full": baseline_full,
           "candidate_focused": report("candidate-focused"), "candidate_full": candidate_full,
           "new_full_suite_failures": sorted(introduced)}
(results / "summary.json").write_text(json.dumps(summary, indent=2))
print("VALIDATION_RESULT=" + json.dumps(summary), flush=True)
