#!/usr/bin/env bash
# Third seed of GRPO at lr 1e-5 -- a spread where there was a difference.
#
# tools/seed_variance.py states the limit of run 42 in its own docstring: two seeds give
# one difference and no variance estimate, so the movement it reports is a lower bound on
# the noise floor. This run makes it three draws -- still a lower bound, but the smallest
# set that can show the dispersion of the thing every verdict in the document is compared
# against. Identical to runs 41 and 42 except for SEED, which the launcher forwards to
# ms-swift and writes into the record, so the three can be shown to differ rather than
# asserted to.
#
# It waits for run 42's landing before training: the landing (and 44_final_consolidation
# behind it) rewrites 29_grpo_train.sh, and bash reads a script by byte offset as it
# executes. Launching while the wrapper is being edited is how two runs were lost here.
#
# The landing is this script's last step, and it enumerates rather than lists: no step
# rebuilds the comparison from a hand-written run list (section 8 item 15), and
# 44_final_consolidation.sh is queued behind this script and rebuilds it from disk.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PY="$VENV/python"
TAG="qwen35_9b_grpo_lr1e5_seed3409"
SEED=3409

step() { printf '\n========== %s ==========\n' "$1"; }

# ------------------------------------------------ 0. wait for the wrapper to be free
step "0. waiting for run 42's landing and the wrapper patches"
waited=0
while : ; do
  if ! pgrep -f "land42_driver|patch75_grpo_record|patch118_grpo_train_fixes" >/dev/null 2>&1; then
    echo "the landing and the wrapper patches are done after ~${waited}s"
    break
  fi
  [ "$waited" -ge 7200 ] && { echo "FATAL: still waiting after ${waited}s" >&2; exit 1; }
  sleep 30
  waited=$((waited + 30))
done
sleep 20

# ------------------------------------------------------------------------- 1. train
step "1. launching $TAG (GRPO G=4, lr 1e-5, seed $SEED)"
cd "$PROJECT" || exit 1
setsid env RUN_TAG="$TAG" \
  IS_LEVEL=token \
  LEARNING_RATE=1e-5 \
  NUM_GENERATIONS=4 \
  GRAD_ACCUM=4 \
  SEED="$SEED" \
  MAX_STEPS=150 \
  SAVE_STEPS=50 \
  GRPO_TIMEOUT=9000 \
  bash scripts/29_grpo_train.sh \
  > "$LOGS/43_${TAG}_outer.log" 2>&1 < /dev/null

TRAIN_RESULT="$REPORTS/${TAG}_train_result.json"
step "2. waiting for $TRAIN_RESULT"
waited=0
until [ -s "$TRAIN_RESULT" ]; do
  [ "$waited" -ge 10800 ] && { echo "FATAL: no train result within ${waited}s" >&2; exit 1; }
  sleep 60
  waited=$((waited + 60))
done
sleep 30
"$PY" -c "
import json
d = json.load(open('$TRAIN_RESULT'))
c = d.get('config', {})
print('  outcome:', d.get('outcome'), '|', d.get('reason'))
print('  steps:', d.get('steps_logged'), '| G:', c.get('num_generations'),
      '| lr:', c.get('learning_rate'), '| seed:', c.get('seed'))
print('  mean_reward:', d.get('mean_reward'), '| mean_kl:', d.get('mean_kl'))
"

# ------------------------------------- 3. prove this seed is a third draw, not a repeat
step "3. seed check against both earlier runs (all three must differ)"
"$PY" - "$REPORTS/grpo_train_result.json" \
        "$REPORTS/qwen35_9b_grpo_lr1e5_seed3408_train_result.json" \
        "$TRAIN_RESULT" <<'PY'
import json, sys
from pathlib import Path
seeds, tags = [], []
for p in sys.argv[1:]:
    d = json.loads(Path(p).read_text(encoding="utf-8")).get("config", {})
    seeds.append(d.get("seed")); tags.append(Path(p).stem)
print("  seeds:", ", ".join(f"{t}={s}" for t, s in zip(tags, seeds)))
if any(s is None for s in seeds):
    print("  WARNING: a record has no seed field; the three cannot be shown to differ")
elif len(set(seeds)) != len(seeds):
    print("  ERROR: two of the seeds are identical -- that is a duplicate, not a draw")
    sys.exit(3)
else:
    print("  all three seeds differ: a third draw, not a copy")
PY

step "4. repoint log, measure idle-step share"
"$PY" - "$TRAIN_RESULT" "$LOGS/24_grpo_${TAG}.log" <<'PY'
import json, re, sys
from pathlib import Path
rec, log = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(rec.read_text(encoding="utf-8"))
if not log.is_file():
    print(f"  WARNING: {log} missing; record left as written")
else:
    vals = [float(v) for v in re.findall(
        r"'frac_reward_zero_std': '([-0-9.eE]+)'",
        log.read_text(encoding="utf-8", errors="replace"))]
    d["log"] = str(log)
    if vals:
        d["mean_frac_reward_zero_std"] = sum(vals) / len(vals)
        d["frac_reward_zero_std_note"] = (
            "Share of logged optimizer steps where every generation in the group "
            "scored identically, so the advantage was zero and no gradient was "
            "produced. Measured over all steps in the per-run log.")
        print(f"  idle steps: {d['mean_frac_reward_zero_std']:.2%} "
              f"({sum(1 for v in vals if v >= 1.0)}/{len(vals)})")
    rec.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  log repointed to {log.name}")
PY

step "5. $TAG : score + retrieval"
RUN_DIR="$ROOT/outputs/checkpoints/$TAG" \
MERGED="$ROOT/models/Qwen3.5-9B-grpo-lr1e5-seed3409-merged" \
NAME="$TAG" RUN_GRPO_RETRIEVAL=1 \
  bash "$PROJECT/scripts/30_eval_grpo.sh" \
  || echo "WARNING: $TAG evaluation returned non-zero; its column may stay partial"

# ------------------------------------------------------------------- 6. land it
step "6. land $TAG (no column; --pre kl_length_confound, seed_variance; --patch patch95_seed3, patch101)"
if bash "$ROOT/tools/land_run.sh" --tag "$TAG" --no-column \
     --pre tools/kl_length_confound.py --pre tools/seed_variance.py \
     --patch tools/patch95_seed3.py --patch tools/patch101_kl_sentences.py; then
  echo "landed"
else
  echo "LANDING STOPPED -- the log above names the sentence or tool that stopped it."
  echo "The comparison is deliberately not rebuilt here: 44_final_consolidation.sh"
  echo "enumerates the run set from disk and is queued behind this script. Rebuilding"
  echo "it from a hand-written list is the defect section 8 item 15 records."
fi

printf '\n========== GRPO seed-3 queue complete ==========\n'
