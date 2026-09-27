"""Patch 29_grpo_train.sh in place: parameterise the algorithm flags.

Run on the server. Three edits:
  1. defaults for IS_LEVEL / LEARNING_RATE / NUM_GENERATIONS / TEMPERATURE
  2. forward those to 24_train_grpo.sh (the launch script reads them)
  3. replace the config block's hardcoded 4 / 1e-5 and its false claim that GSPO
     is unavailable -- it is available as --importance_sampling_level sequence,
     which is what the run now records.

Each replacement is asserted to have happened exactly once, so a silently
missed edit fails loudly instead of leaving the script half-patched.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/29_grpo_train.sh")
src = path.read_text(encoding="utf-8")

edits = [
    # 1. defaults, right after the timeout budget
    ('MAX_STEPS="${MAX_STEPS:-150}"',
     'MAX_STEPS="${MAX_STEPS:-150}"\n'
     '# Algorithm flags. `sequence` importance sampling is GSPO; the ms-swift\n'
     '# default is `token`, which is plain GRPO. Kept as variables so the two\n'
     '# runs differ by a documented flag rather than by an edited script.\n'
     'IS_LEVEL="${IS_LEVEL:-token}"\n'
     'LEARNING_RATE="${LEARNING_RATE:-1e-5}"\n'
     'NUM_GENERATIONS="${NUM_GENERATIONS:-4}"\n'
     'TEMPERATURE="${TEMPERATURE:-1.0}"'),
    # 2. forward them to the launch script
    ('SMOKE=0 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \\\n'
     '  timeout "$TIMEOUT" bash "$PROJECT/scripts/24_train_grpo.sh" > "$LOG" 2>&1',
     'SMOKE=0 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \\\n'
     '  RUN="${RUN_TAG:-grpo}" IS_LEVEL="$IS_LEVEL" LEARNING_RATE="$LEARNING_RATE" \\\n'
     '  NUM_GENERATIONS="$NUM_GENERATIONS" TEMPERATURE="$TEMPERATURE" \\\n'
     '  timeout "$TIMEOUT" bash "$PROJECT/scripts/24_train_grpo.sh" > "$LOG" 2>&1'),
    # 3. the config block records what actually ran
    ('        "num_generations": 4,', '        "num_generations": int(os.environ.get("NUM_GENERATIONS", 4)),'),
    ('        "learning_rate": 1e-5,', '        "learning_rate": float(os.environ.get("LEARNING_RATE", 1e-5)),'),
    ('        "gspo": "not offered by this ms-swift commit; rlhf_type choices exclude gspo",',
     '        "importance_sampling_level": os.environ.get("IS_LEVEL", "token"),\n'
     '        "algorithm": ("GSPO (sequence-level importance sampling)"\n'
     '                      if os.environ.get("IS_LEVEL") == "sequence" else "GRPO"),\n'
     '        "gspo_note": ("GSPO is not a separate rlhf_type in ms-swift; it is GRPO with "\n'
     '                      "--importance_sampling_level sequence. An earlier revision of "\n'
     '                      "this file claimed GSPO was unavailable because the rlhf_type "\n'
     '                      "choices exclude the name gspo -- that was wrong."),'),
    # `os` must be imported for the two env lookups above
    ('import json, re, sys', 'import json, os, re, sys'),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        sys.exit(f"FAILED: expected exactly 1 occurrence, found {count}:\n{old[:120]}")
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("patched OK; all 6 replacements applied exactly once")
