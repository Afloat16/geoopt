"""Install the locked dev packages and a pinned CPU Torch through Kornia's Pixi environment."""
import os
from pathlib import Path
import subprocess
import tomllib

packages = tomllib.loads(Path("uv.lock").read_text())["package"]
sidecars = sorted({
    package["name"] for package in packages
    if package["name"] in {"torch", "triton"} or package["name"].startswith(("nvidia-", "cuda-"))
})
command = ["uv", "sync", "--locked", "--extra", "dev"]
for name in sidecars:
    command.extend(["--no-install-package", name])
subprocess.run(command, check=True)
subprocess.run([
    "uv", "pip", "install", "--python", ".venv-py312/bin/python",
    "--index-url", "https://download.pytorch.org/whl/cpu", "torch==" + os.environ["TORCH_VERSION"],
], check=True)
subprocess.run([
    str(Path(".venv-py312/bin/python").absolute()), ".github/scripts/check_torch_env.py",
    "--channel", "pinned", "--expected", os.environ["TORCH_VERSION"],
], check=True)
