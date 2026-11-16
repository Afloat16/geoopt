import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
payload = json.loads(pathlib.Path(__file__).with_name("pml-manifold-payload.json").read_text())
scopes = {
    "policy": {".code-polishy.json", "docs/security/dependency-assessments.md"},
    "tests": {"tests/losses/test_manifold_precision.py"},
    "all": set(payload),
}
for path in scopes[sys.argv[2]]:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload[path])
