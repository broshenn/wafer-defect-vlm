#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_d.sh BASE_MODEL D_ADAPTER GPU_UUID NEW_RECORDS_DIRECTORY
if [ "$#" -ne 4 ]; then
  echo 'Usage: bash run_d.sh BASE_MODEL D_ADAPTER GPU_UUID NEW_RECORDS_DIRECTORY' >&2
  exit 2
fi
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_model="$1"
d_adapter="$2"
gpu_uuid="$3"
records_dir="$4"
case "$gpu_uuid" in GPU-*) ;; *) echo 'An authorized GPU UUID is required.' >&2; exit 2 ;; esac
test -d "$base_model"
test -d "$d_adapter"
if [ -e "$records_dir" ]; then
  echo 'Use a new records directory; existing output is preserved.' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export IMAGE_MAX_TOKEN_NUM=256 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export WAFER_CHECKER="$bundle_dir/schema_check.py"
python "$bundle_dir/verify_bundle.py" --adapter "$d_adapter"
python "$bundle_dir/test_cpu_contract.py"
exec python "$bundle_dir/wafer_service.py" \
  --model "$base_model" --adapter "$d_adapter" --tag D-N3072-3407 \
  --prompt "$bundle_dir/prompt_7f.txt" --checker "$bundle_dir/schema_check.py" \
  --out "$records_dir" --host 127.0.0.1 --port 7860 --max-runtime-seconds 2700
