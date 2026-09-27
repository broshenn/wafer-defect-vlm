"""Make the training seed overridable, and record it.

All five RL runs so far used the same seed, so the project has no measurement of
run-to-run variance, and the headline conclusion is a negative one ("no RL
config beats SFT") that rests entirely on confidence intervals computed within
single runs. A second seed of the strongest RL config is the cheapest way to
test whether those intervals reflect the comparison or only sampling noise.

Three edits, and the third matters most: the run record must carry the seed, or
there is no way to prove two runs differ. This project has already shipped twice
with something that looked parameterised and was not (LIMITATIONS section 8,
items 5 and 6), so the seed is written into the record rather than trusted to
the shell variable name.
"""
import ast
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
SCRIPTS = ROOT / "projects/wafer-defect-vlm/scripts"
edits = []

# ---- 1. the inner script takes the seed from the environment
p24 = SCRIPTS / "24_train_grpo.sh"
s24 = p24.read_text(encoding="utf-8")
old24 = "  --seed 3407\n"
new24 = ('  --seed "${SEED:-3407}"\n')
edits.append(("24:seed-flag", p24, s24, old24, new24))

# ---- 2 and 3. the launcher forwards it and records it
p29 = SCRIPTS / "29_grpo_train.sh"
s29 = p29.read_text(encoding="utf-8")

old29a = '''  NUM_GENERATIONS="$NUM_GENERATIONS" TEMPERATURE="$TEMPERATURE" \\
  timeout "$TIMEOUT" bash "$PROJECT/scripts/24_train_grpo.sh" > "$LOG" 2>&1'''
new29a = '''  NUM_GENERATIONS="$NUM_GENERATIONS" TEMPERATURE="$TEMPERATURE" \\
  SEED="${SEED:-3407}" \\
  timeout "$TIMEOUT" bash "$PROJECT/scripts/24_train_grpo.sh" > "$LOG" 2>&1'''
edits.append(("29:forward", p29, s29, old29a, new29a))

old29b = '''        "temperature": 1.0,
        "reward_funcs": ["wafer_class", "wafer_format", "wafer_radial", "wafer_clock"],'''
new29b = '''        "temperature": 1.0,
        "seed": int(os.environ.get("SEED", 3407)),
        "reward_funcs": ["wafer_class", "wafer_format", "wafer_radial", "wafer_clock"],'''
edits.append(("29:record", p29, s29, old29b, new29b))

# apply, with each file written once
by_file = {}
for tag, path, src, old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED at {tag}: {n} occurrences (need exactly 1)")
    by_file.setdefault(path, src)
    by_file[path] = by_file[path].replace(old, new)
    print(f"  {tag}: ok")

for path, src in by_file.items():
    if path.suffix == ".py":
        ast.parse(src)
    path.write_text(src, encoding="utf-8")
print("seed is now overridable and is written into the run record")
