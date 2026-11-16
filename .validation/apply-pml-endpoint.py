import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
name, scope = sys.argv[2:4]
payload = json.loads(pathlib.Path(__file__).with_name("pml-" + name + ".json").read_text())
if scope == "format":
    subprocess.run(["./code-polishyw", "format", "--files", *payload], cwd=root, check=True)
    subprocess.run(["git", "add", "--", *payload], cwd=root, check=True)
    print("FORMATTED_PAYLOAD=" + json.dumps({p: (root / p).read_text() for p in payload}))
else:
    scopes = {
        "policy": {".code-polishy.json", "docs/security/dependency-assessments.md"},
        "tests": {p for p in payload if p.startswith("tests/")},
        "all": set(payload),
    }
    for path in scopes[scope]:
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload[path])
