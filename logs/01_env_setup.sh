#!/usr/bin/env bash
set -uo pipefail
F=/root/autodl-fs/wafer-vlm
VENV=$F/venvs/wafer
PY=/root/miniconda3/bin/python
MS_SWIFT_COMMIT=9d3d03dd35c8a60e519e3165962586837cea67ee

step() { echo; echo "########## $* ##########"; }

step "create venv"
$PY -m venv $VENV || exit 1
source $VENV/bin/activate
python -V
pip install -U pip setuptools wheel || exit 1

step "torch cu128 (pinned)"
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 || exit 1

step "project + deps"
cd $F/projects/wafer-defect-vlm
pip install -e ".[dev,quality,train]" || exit 1

step "bitsandbytes (QLoRA)"
pip install bitsandbytes || echo "WARN bitsandbytes failed"

step "ms-swift pinned commit (via github proxy)"
set +u
source /etc/network_turbo
set -u
pip install "git+https://github.com/modelscope/ms-swift.git@${MS_SWIFT_COMMIT}" || echo "WARN ms-swift install failed"
unset http_proxy https_proxy

step "freeze"
pip freeze > $F/manifests/venv_locked_freeze.txt

step "verify"
python - <<'PY'
import importlib.util, torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for m in ("swift", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "wafer_vlm", "pandas", "pyarrow"):
    spec = importlib.util.find_spec(m)
    print(m, "OK" if spec else "MISSING")
import transformers, pandas
print("transformers", transformers.__version__)
print("pandas", pandas.__version__)
PY

echo
echo "ENV_DONE $(date)"
