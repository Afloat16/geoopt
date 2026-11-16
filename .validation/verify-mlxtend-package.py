import json
from pathlib import Path
import sys

name = sys.argv[1]
records = json.loads(Path(".validation/mlxtend-first-formatted.json").read_text())
record = next(x for x in records if x["name"] == name)
project = Path("project")
for path, expected in record["files"].items():
    assert (project / path).read_text() == expected, path
print("EXACT_VALIDATED_FILES_MATCH", name, flush=True)
