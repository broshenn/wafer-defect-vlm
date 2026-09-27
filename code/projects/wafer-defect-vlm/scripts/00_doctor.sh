#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
PROJECT_DIR="$ROOT_DIR/projects/wafer-defect-vlm"
source "$ROOT_DIR/venvs/wafer/bin/activate"

python - <<'PY'
import importlib.util
import platform
import shutil
import sys

print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
for package in ("swift", "torch", "transformers", "wafer_vlm"):
    spec = importlib.util.find_spec(package)
    print(package, "OK" if spec else "MISSING")
print("swift_cli", shutil.which("swift"))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
df -h "$ROOT_DIR"
test -d "$PROJECT_DIR"
