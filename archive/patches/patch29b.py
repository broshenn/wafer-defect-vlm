"""Second pass on 29_grpo_train.sh: make the output paths per-run.

The first patch missed these three lines, so the GSPO run overwrote the GRPO
run's record -- the exact collision the surrounding comment warns about. The
GRPO record survived only because it was already committed, which is luck, not
design. Parameterised here so a third run cannot repeat it.

Also stamps `algorithm` onto both existing records, since the GRPO one predates
the discovery that GSPO is reachable via --importance_sampling_level sequence.
"""
import json
import sys
from pathlib import Path

script = Path("/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/29_grpo_train.sh")
src = script.read_text(encoding="utf-8")

edits = [
    ('RESULT="$ROOT/outputs/reports/grpo_train_result.json"',
     'RUN_TAG="${RUN_TAG:-grpo}"\n'
     'RESULT="$ROOT/outputs/reports/${RUN_TAG}_train_result.json"'),
    ('LOG="$ROOT/logs/29_grpo_train.log"', 'LOG="$ROOT/logs/29_${RUN_TAG}_train.log"'),
    ('MEMLOG="$ROOT/logs/29_grpo_mem.log"', 'MEMLOG="$ROOT/logs/29_${RUN_TAG}_mem.log"'),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"FAILED: {src.count(old)} occurrences of {old!r}")
    src = src.replace(old, new)
script.write_text(src, encoding="utf-8")
print("29_grpo_train.sh: output paths now per-run")

# The first patch wrote the result to the old hardcoded name, so the GSPO record
# currently sits at grpo_train_result.json only in the archived copy; make sure
# both files carry an honest algorithm label.
stamps = {
    "grpo_train_result.json": (
        "GRPO (token-level importance sampling)",
        "This run is plain GRPO, not GSPO. GSPO in ms-swift is GRPO with "
        "--importance_sampling_level sequence, a separate flag in the same args "
        "class; an earlier revision of this record wrongly claimed GSPO was "
        "unavailable because the rlhf_type choices exclude the name gspo."),
    "qwen35_9b_gspo_v1_train_result.json": (
        "GSPO (sequence-level importance sampling)",
        "GSPO run: --importance_sampling_level sequence, lr 5e-5, "
        "num_generations 8 (the paper's 32 does not fit this environment without "
        "vLLM). Everything else matches the GRPO run."),
}
for name, (algorithm, note) in stamps.items():
    path = Path("/root/autodl-fs/wafer-vlm/outputs/reports") / name
    if not path.is_file():
        print(f"  skip (absent): {name}")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.setdefault("config", {})
    cfg["algorithm"] = algorithm
    cfg["algorithm_note"] = note
    if "gspo" in cfg:  # the false "not offered by this ms-swift commit" claim
        del cfg["gspo"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  stamped {name}: {algorithm}")
