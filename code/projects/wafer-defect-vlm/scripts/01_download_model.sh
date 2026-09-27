#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/Qwen3.5-9B}"
source "$ROOT_DIR/venvs/wafer/bin/activate"
mkdir -p "$MODEL_DIR" "$ROOT_DIR/cache/modelscope"

export MODELSCOPE_CACHE="$ROOT_DIR/cache/modelscope"
modelscope download "$MODEL_ID" --repo-type model --local-dir "$MODEL_DIR" --max-workers 4
python - "$MODEL_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [root / "config.json", root / "tokenizer_config.json"]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"incomplete model download, missing: {missing}")
files = sorted(path for path in root.rglob("*") if path.is_file())
manifest = {
    "model_dir": str(root.resolve()),
    "files": len(files),
    "total_bytes": sum(path.stat().st_size for path in files),
    "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
}
(root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
