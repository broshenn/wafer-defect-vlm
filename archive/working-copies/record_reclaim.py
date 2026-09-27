"""Record the disk reclaim, and the preconditions that made it safe.

Deleting 108 GB of merged models is the kind of action that should leave a trace
saying what was removed, what was checked first, and what remains -- otherwise a
later reader finds a path that a report or a tool refers to and cannot tell
whether it was never there or was deliberately removed.

The record is appended to, not overwritten, so several reclaims in a day
accumulate rather than the last one erasing the others.
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
OUT = R / "outputs/reports/disk_reclaim.json"

REMOVED = [
    ("qwen35_9b_qlora_v1/v0-20260915-135822/checkpoint-604-merged",
     "qwen35_9b_adapter", "merged_rankings.jsonl"),
    ("qwen35_9b_grpo_v1/v0-20260915-171503/checkpoint-150-merged",
     "qwen35_9b_grpo", "grpo_rankings.jsonl"),
    ("qwen35_9b_gspo_v1/v0-20260915-191257/checkpoint-150-merged",
     "qwen35_9b_gspo_v1", "qwen35_9b_gspo_v1_rankings.jsonl"),
    ("gspo_lr1e5/v0-20260915-201919/checkpoint-150-merged",
     "gspo_lr1e5", "gspo_lr1e5_rankings.jsonl"),
    ("qwen35_9b_gspo_g32/v0-20260915-213224/checkpoint-150-merged",
     "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g32_rankings.jsonl"),
]
EACH_MB = 17968

entry = {
    "when": datetime.now().isoformat(timespec="seconds"),
    "why": ("Each run left an ~18 GB bf16 merge inside its checkpoint directory. "
            "Queues 41 and 42 each still need ~26 GB for their own merge and "
            "checkpoints against 50 GB free, so the second would have run out of "
            "disk partway through. Merged models are intermediates; the basis for "
            "treating them as reclaimable is below."),
    "preconditions_checked_per_run": {
        "rankings_file_present_and_nonempty": (
            "the retrieval metric is computed from outputs/retrieval/<name>_rankings.jsonl, "
            "so with that file present the recorded mAP@10 stays reproducible with no "
            "merged model on disk"),
        "adapter_checkpoint_present": (
            "the merge is regenerable from the LoRA adapter (~5 min); all five "
            "adapter_model.safetensors were present at 173.2 MB"),
        "recorded_mAP@10_present": (
            "the report for each run already held a retrieval mAP@10, so no "
            "unrecorded measurement depended on the directory"),
    },
    "removed": [
        {"path": f"outputs/checkpoints/{p}", "approx_mb": EACH_MB,
         "report": f"{tag}__report.json", "rankings_kept": rf}
        for p, tag, rf in REMOVED],
    "removed_total_gb": round(EACH_MB * len(REMOVED) / 1024, 2),
    "kept_deliberately": [
        {"path": "outputs/checkpoints/qwen35_9b_grpo_lr5e5/"
                 "v0-20260916-121847/checkpoint-150-merged",
         "why": "driver 40 was writing and reading it at the time; excluded explicitly"},
        {"path": "models/Qwen3.5-9B-sft-lora-v1-merged",
         "why": "smaller models/ copy kept for the SFT run"},
        {"path": "models/Qwen3.5-9B-grpo-v1-merged",
         "why": "smaller models/ copy kept for the GRPO lr1e-5 run"},
    ],
    "free_space": {"before": "45G", "after": "133G"},
    "not_resolved": (
        "The models/ merged copies are 8.55 GB with 2000 weight-map keys while the "
        "checkpoint copies were 18.82 GB with 760 keys, both declaring bfloat16. "
        "That does not reconcile and was not explained. It did not block this "
        "reclaim, which rests on the rankings files and the adapters rather than on "
        "either copy being canonical -- but anyone who needs a merged model should "
        "re-merge from the adapter rather than assume the models/ copy is complete."),
}

history = []
if OUT.is_file():
    try:
        history = json.loads(OUT.read_text(encoding="utf-8")).get("reclaims", [])
    except json.JSONDecodeError:
        history = []
history.append(entry)
OUT.write_text(json.dumps({"reclaims": history}, ensure_ascii=False, indent=2),
               encoding="utf-8")
print(f"recorded {len(REMOVED)} removals in {OUT.relative_to(R)}"
      f" ({len(history)} reclaim event(s) total)")

free = shutil.disk_usage("/root/autodl-fs").free / 1e9
print(f"filesystem free now: {free:.1f} GB")
