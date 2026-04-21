"""Modal app wrapper — reads deploy.toml and serves the project."""

import modal
import tomllib
from pathlib import Path

# Read deploy config
config_path = Path(__file__).parent / "deploy.toml"
with open(config_path, "rb") as f:
    config = tomllib.load(f)

deploy = config["deploy"]
limits = config["deploy"]["limits"]
entrypoint = deploy["entrypoint"]
mode = deploy.get("mode", "static")

app = modal.App("housing-analyzer")

# Build image: install project dependencies via uv
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir(".", "/root/project", copy=True, ignore=[".venv", ".git", "__pycache__", ".github", "node_modules"])
    .run_commands("cd /root/project && uv sync --no-dev")
)


@app.function(
    image=image,
    cpu=limits.get("cpu", 2),
    memory=limits.get("memory", 512),
    timeout=limits.get("timeout", 300),
    scaledown_window=limits.get("idle_timeout", 300),
)
@modal.web_server(8080)
def serve():
    import subprocess

    entrypoint_path = f"/root/project/{entrypoint}"

    if mode == "serve":
        subprocess.Popen(
            ["uv", "run", "marimo", "run", entrypoint_path,
             "--host", "0.0.0.0", "--port", "8080", "--no-token"],
            cwd="/root/project",
        )
    else:
        subprocess.Popen(
            ["uv", "run", "python", "-m", "http.server", "8080",
             "--directory", "/root/project/dist"],
            cwd="/root/project",
        )
