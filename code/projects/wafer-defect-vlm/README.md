# Wafer Defect VLM Reproduction

This repository implements the reproducible path from WM811K BIN matrices to a reviewed benchmark and Qwen3.5-9B multimodal post-training with Alibaba `ms-swift`.

The canonical server root is `/root/autodl-tmp/wafer-vlm`. Raw data is immutable; all generated images, annotations, curated datasets, benchmark files, checkpoints, and reports are written to separate directories.

## 1. Environment check

```bash
cd /root/autodl-tmp/wafer-vlm/projects/wafer-defect-vlm
bash scripts/00_doctor.sh
```

Do not load `LSWMD.pkl` in the current no-GPU 2 GB container. Switch to a GPU instance with adequate system RAM first.

## 2. Prepare images and deterministic geometry

The default subset contains at most 800 samples for each defect class and 400 `none` samples. Splitting is by `lot_name` before task expansion.

```bash
source /root/autodl-tmp/wafer-vlm/venvs/wafer/bin/activate
cd /root/autodl-tmp/wafer-vlm/projects/wafer-defect-vlm
python -m wafer_vlm.prepare_data \
  --input /root/autodl-tmp/wafer-vlm/data/raw/wm811k/LSWMD.pkl \
  --output /root/autodl-tmp/wafer-vlm/data/prepared_v1 \
  --per-class 800 \
  --none-limit 400 \
  --resolution 448 \
  --seed 3407
```

Immediately construct the draft benchmark from the untouched test split. It is deliberately marked pending until two-person review is recorded.

```bash
python -m wafer_vlm.benchmark build \
  --manifest /root/autodl-tmp/wafer-vlm/data/prepared_v1/manifest.jsonl \
  --output /root/autodl-tmp/wafer-vlm/benchmarks/wafer_bench_v1 \
  --per-class 30 \
  --queries-per-class 10
```

## 3. Teacher selection and annotation

Secrets are loaded only from `/root/autodl-tmp/wafer-vlm/secrets/api.env`. Never print or commit that file.

First compare both teachers on the same 20–30 training samples per class. Use identical prompts and parameters. Human review decides whether the cheaper teacher is acceptable.

```bash
ENV=/root/autodl-tmp/wafer-vlm/secrets/api.env
MANIFEST=/root/autodl-tmp/wafer-vlm/data/prepared_v1/manifest.jsonl
OUT=/root/autodl-tmp/wafer-vlm/data/annotations

python -m wafer_vlm.annotate --manifest "$MANIFEST" --output "$OUT/selection_deepseek.jsonl" \
  --env-file "$ENV" --provider dashscope --model deepseek-v4.1-flash \
  --split train --per-class 25 --concurrency 4
python -m wafer_vlm.annotate --manifest "$MANIFEST" --output "$OUT/selection_qwen397b.jsonl" \
  --env-file "$ENV" --provider dashscope --model qwen3.5-397b-a17b \
  --split train --per-class 25 --concurrency 4
```

After teacher acceptance, annotate train and validation only with DeepSeek. Then select all failed/boundary samples plus a stable 15% audit for Qwen3.5-397B.

```bash
python -m wafer_vlm.annotate --manifest "$MANIFEST" --output "$OUT/deepseek_primary.jsonl" \
  --env-file "$ENV" --provider dashscope --model deepseek-v4.1-flash --split train
python -m wafer_vlm.annotate --manifest "$MANIFEST" --output "$OUT/deepseek_primary.jsonl" \
  --env-file "$ENV" --provider dashscope --model deepseek-v4.1-flash --split val

python -m wafer_vlm.curate plan --manifest "$MANIFEST" --primary "$OUT/deepseek_primary.jsonl" \
  --output "$OUT/primary_quality.jsonl" --ids-output "$OUT/qwen_audit_ids.txt" \
  --audit-fraction 0.15

python -m wafer_vlm.annotate --manifest "$MANIFEST" --output "$OUT/qwen_secondary.jsonl" \
  --env-file "$ENV" --provider dashscope --model qwen3.5-397b-a17b --split train \
  --ids-file "$OUT/qwen_audit_ids.txt"
python -m wafer_vlm.annotate --manifest "$MANIFEST" --output "$OUT/qwen_secondary.jsonl" \
  --env-file "$ENV" --provider dashscope --model qwen3.5-397b-a17b --split val \
  --ids-file "$OUT/qwen_audit_ids.txt"

python -m wafer_vlm.curate finalize --manifest "$MANIFEST" \
  --primary "$OUT/deepseek_primary.jsonl" --secondary "$OUT/qwen_secondary.jsonl" \
  --output-dir /root/autodl-tmp/wafer-vlm/data/curated_v1
```

Teacher disagreement on core fields is quarantined instead of silently choosing one answer. Test samples never enter teacher annotation or SFT output.

## 4. Model and baselines

```bash
bash scripts/01_download_model.sh
bash scripts/20_run_benchmark.sh
```

The zero-shot benchmark must be completed and saved before training. The benchmark cannot be officially frozen until every core item has two-person approval:

```bash
python -m wafer_vlm.benchmark freeze \
  --output /root/autodl-tmp/wafer-vlm/benchmarks/wafer_bench_v1
```

## 5. Post-training

Use QLoRA first on a 24 GB-class GPU. On a larger GPU, run BF16 LoRA. Do not start visual-side training until the language-side baseline has been evaluated.

```bash
bash scripts/10_train_qlora.sh
# or, when memory permits:
bash scripts/11_train_lora_bf16.sh

ADAPTER_DIR=/root/autodl-tmp/wafer-vlm/outputs/checkpoints/<run>/<checkpoint> \
  bash scripts/21_run_adapter_benchmark.sh
```

The optional `scripts/12_train_vision_lora.sh` uses `lora_llm`: LoRA on the LLM and full training on the vision/aligner side with a lower learning rate. Run it only after the frozen-ViT baseline proves that visual recognition is the bottleneck.

## Acceptance gates

- Prepared manifest IDs and rendered image names are unique.
- Train/validation/test lot intersections are empty.
- API output remains traceable to provider, model, prompt version, and deterministic validation.
- Benchmark remains test-only, reviewed, hashed, and absent from training/prompt tuning.
- Complete zero-shot and adapter inference use the same benchmark requests.
- Every run records model/data hashes, fixed seed, dependency versions, GPU model, and exact command.
