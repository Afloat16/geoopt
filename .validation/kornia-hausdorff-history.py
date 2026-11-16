"""Read-only public history inventory. It does not import Kornia or run tests."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = "b1624c022ceb93373cb073a50bd1da2511108c5e"
OUT = Path.cwd().parent / "kornia-hausdorff-history-results"
OUT.mkdir(exist_ok=True)

def git(*args):
    return subprocess.check_output(["git", *args], text=True)

assert git("rev-parse", "HEAD").strip() == BASE
ref_lines = git("for-each-ref", "--format=%(objectname) %(refname)").splitlines()
refs = [{"sha": x.split(" ", 1)[0], "ref": x.split(" ", 1)[1]} for x in ref_lines]
(OUT / "all-refs.json").write_text(json.dumps(refs, indent=2) + "\n")

# Follow rename chains in both directions, including merge-parent differences.
change_lines = git("log", "--all", "--full-history", "-m", "--format=", "--name-status", "--find-renames").splitlines()
all_paths = set()
renames = []
for line in change_lines:
    parts = line.split("\t")
    if len(parts) >= 2:
        all_paths.update(parts[1:])
    if len(parts) == 3 and parts[0].startswith("R"):
        renames.append(parts[1:])

aliases = {"kornia/losses/hausdorff.py", "torchgeometry/losses/hausdorff.py"}
aliases.update(path for path in all_paths if "/losses/" in path and "hausdorff" in path.lower() and path.endswith(".py"))
# A class moved into a differently named file can evade basename-only searches.
definitions = git(
    "log", "--all", "--full-history", "--format=", "--name-only",
    "-G", "class[[:space:]]+[^ :]*Hausdorff|def[[:space:]]+[^ (]*hausdorff", "--", "*.py"
).splitlines()
aliases.update(path for path in definitions if path.endswith(".py"))
changed = True
while changed:
    changed = False
    for old, new in renames:
        if (old in aliases or new in aliases) and not {old, new}.issubset(aliases):
            aliases.update((old, new))
            changed = True

object_lines = git("rev-list", "--objects", "--all", "--", *sorted(aliases)).splitlines()
objects = {}
for line in object_lines:
    parts = line.split(" ", 1)
    if len(parts) == 2 and parts[1] in aliases:
        objects.setdefault(parts[0], set()).add(parts[1])

source_records = []
variants = {}
parse_failures = []
for oid, paths in sorted(objects.items()):
    if git("cat-file", "-t", oid).strip() != "blob":
        continue
    source = git("cat-file", "blob", oid)
    record = {"blob": oid, "paths": sorted(paths), "sha256": hashlib.sha256(source.encode()).hexdigest()}
    try:
        parsed = ast.parse(source)
    except SyntaxError as error:
        record["parse_error"] = str(error)
        record["source"] = source
        parse_failures.append(record)
        source_records.append(record)
        continue
    targets = [
        node for node in ast.walk(parsed)
        if isinstance(node, ast.ClassDef) and "Hausdorff" in node.name
        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "hausdorff" in node.name.lower()
    ]
    if not targets:
        record["contains_target"] = False
        source_records.append(record)
        continue
    record["contains_target"] = True
    record["source"] = source
    record["keywords"] = [
        {"line": i, "text": line}
        for i, line in enumerate(source.splitlines(), 1)
        if re.search(r"float16|bfloat16|float32|promot|dtype|perform_erosion|weight\\s*=", line, re.I)
    ]
    for node in targets:
        # Ignore docstrings, but retain every executable node and literal.
        for child in ast.walk(node):
            if isinstance(child, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.body and isinstance(child.body[0], ast.Expr) and isinstance(child.body[0].value, ast.Constant) and isinstance(child.body[0].value.value, str):
                    child.body = child.body[1:]
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        version = variants.setdefault(digest, {"name": node.name, "source": ast.get_source_segment(source, node), "blobs": []})
        version["blobs"].append(oid)
        record.setdefault("variant_digests", []).append(digest)
    source_records.append(record)

report = {
    "base": BASE,
    "scope": "All fetched public upstream branches/tags/PR heads and all Afloat16 fork branch heads; every reachable intermediate source object, rename closure, and historical Hausdorff loss definitions.",
    "ref_count": len(refs),
    "upstream_pr_refs": sum(x["ref"].startswith("refs/remotes/upstream-pr/") for x in refs),
    "fork_refs": sum(x["ref"].startswith("refs/remotes/user-fork/") for x in refs),
    "aliases": sorted(aliases),
    "source_blob_count": len(source_records),
    "target_source_blob_count": sum(x.get("contains_target", False) for x in source_records),
    "variant_count": len(variants),
    "parse_failure_count": len(parse_failures),
    "records": source_records,
    "variants": variants,
    "parse_failures": parse_failures,
    "known_excluded_direction": "CPU class batching was already proposed in https://github.com/kornia/kornia/pull/6386; do not repeat it.",
    "classification": "Evidence inventory only: inspect variants before asserting novelty. No product source was changed or imported."
}
raw = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
(OUT / "hausdorff-history.json").write_bytes(raw)
encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
summary = {k: v for k, v in report.items() if k not in ("records", "variants", "parse_failures")}
summary.update({"report_sha256": hashlib.sha256(raw).hexdigest(), "report_bytes": len(raw), "base64_chunks": (len(encoded) + 4095) // 4096})
print("AUDIT_SUMMARY " + json.dumps(summary), flush=True)
for index, offset in enumerate(range(0, len(encoded), 4096)):
    print(f"AUDIT_CHUNK {index:04d} {encoded[offset:offset+4096]}", flush=True)
