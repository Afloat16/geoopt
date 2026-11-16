import ast
import hashlib
import io
import json
import pathlib
import tomllib
import urllib.request
import zipfile
root = pathlib.Path("policy-source")
lock = tomllib.loads((root / "uv.lock").read_text())
packages = {p["name"]: p for p in lock["package"]}
material = packages["mkdocs-material"]
wheel = material["wheels"][0]
with urllib.request.urlopen(wheel["url"]) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == wheel["hash"].split(":")[1]
archive = zipfile.ZipFile(io.BytesIO(data))
wheel_hits = {}
for name in archive.namelist():
    if name.endswith(".py"):
        text = archive.read(name).decode("utf-8")
        hits = [f"{i+1}: {line}" for i, line in enumerate(text.splitlines()) if any(x in line for x in ("requests", "urllib3", "certifi"))]
        if hits:
            wheel_hits[name] = hits
repo_hits = {}
for scope in ("src", "tests", "scripts", "benchmarks", "docs/recipes"):
    for path in (root / scope).rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [x.name for x in node.names if x.name.split(".")[0] in {"requests", "urllib3", "certifi"}]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in {"requests", "urllib3", "certifi"}:
                imports.append(node.module)
        if imports:
            repo_hits[str(path.relative_to(root))] = imports
config = (root / "mkdocs.yml").read_text()
print(json.dumps({"material_version": material["version"], "wheel_url": wheel["url"], "wheel_hash_verified": True, "wheel_https_clients": wheel_hits, "repository_client_imports": repo_hits, "config": config, "material_theme_defaults": archive.read("material/mkdocs_theme.yml").decode()}, indent=2))
