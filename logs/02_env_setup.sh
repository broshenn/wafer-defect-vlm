#!/usr/bin/env bash
# Build the wafer-vlm venv.
# NOTE: built on the local instance disk (/root/autodl-tmp) because the
# /root/autodl-fs FUSE mount creates only ~44 files/s (measured 2026-09-15),
# which makes a ~45k-file venv impractical. The pinned freeze + this script are
# persisted under /root/autodl-fs/wafer-vlm/manifests/, and
# /root/autodl-fs/wafer-vlm/venvs/wafer is a symlink to the built venv.
set -uo pipefail
F=/root/autodl-fs/wafer-vlm
VENV=/root/autodl-tmp/wafer-vlm/venvs/wafer
SRC=/root/miniconda3/lib/python3.12/site-packages
MSSWIFT=/root/autodl-tmp/wafer-vlm/src/ms-swift
PROJ=$F/projects/wafer-defect-vlm
export PIP_DISABLE_PIP_VERSION_CHECK=1
step(){ echo; echo "########## $* ##########"; date "+%F %T"; }

step "clean previous attempt"
rm -rf "$VENV" "$F/venvs/wafer"
mkdir -p "$(dirname "$VENV")"

step "create venv"
/root/miniconda3/bin/python -m venv "$VENV" || exit 1
VP="$VENV/lib/python3.12/site-packages"
"$VENV/bin/python" -V || exit 1

step "seed torch stack from miniconda (avoids ~4GB download)"
for e in functorch nvidia torch torchgen torchvision torchvision.libs triton \
         torch-2.8.0+cu128.dist-info torchvision-0.23.0+cu128.dist-info triton-3.4.0.dist-info; do
  if [ -e "$SRC/$e" ]; then cp -a "$SRC/$e" "$VP/" && echo "  copied $e"; else echo "  MISSING $e"; fi
done
for d in "$SRC"/nvidia_*_cu12-*.dist-info; do cp -a "$d" "$VP/" 2>/dev/null; done
echo "  nvidia dist-info: $(ls -d "$VP"/nvidia_*_cu12-*.dist-info 2>/dev/null | wc -l)"

step "bootstrap pip"
"$VENV/bin/python" -m pip install -U pip setuptools wheel || exit 1

step "torch small runtime deps (copied dirs carry only the big binaries)"
"$VENV/bin/python" -m pip install "typing-extensions>=4.10.0" filelock fsspec "jinja2>=3.1" "networkx>=2.5" "sympy>=1.13.3" "numpy>=1.26,<3" "pillow>=10" || exit 1

step "torch sanity"
"$VENV/bin/python" -c "import torch,torchvision;print(\"torch\",torch.__version__,\"tv\",torchvision.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))" || exit 1

step "ms-swift editable at pinned commit 9d3d03dd35c8a60e519e3165962586837cea67ee"
"$VENV/bin/python" -m pip install -e "$MSSWIFT" || echo "WARN ms-swift install returned non-zero"

step "project editable with extras"
cd "$PROJ" && "$VENV/bin/python" -m pip install -e ".[dev,quality,train]" || echo "WARN project install returned non-zero"

step "bitsandbytes"
"$VENV/bin/python" -m pip install bitsandbytes || echo "WARN bitsandbytes failed"

step "freeze"
"$VENV/bin/python" -m pip freeze > "$F/manifests/venv_locked_freeze.txt"
wc -l "$F/manifests/venv_locked_freeze.txt"

step "symlink documented path"
ln -sfn "$VENV" "$F/venvs/wafer"
ls -la "$F/venvs/"

step "verify"
"$VENV/bin/python" - <<"PY"
import importlib.util, torch, transformers, pandas
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for m in ("swift","peft","trl","accelerate","datasets","bitsandbytes","modelscope",
          "wafer_vlm","pandas","pyarrow","sklearn","cv2","scipy","PIL","safetensors"):
    print(f"  {m:12s}", "OK" if importlib.util.find_spec(m) else "MISSING")
print("transformers", transformers.__version__)
print("pandas", pandas.__version__)
PY
echo
echo "ENV_DONE $(date "+%F %T")"
