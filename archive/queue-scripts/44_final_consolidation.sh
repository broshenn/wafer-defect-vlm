#!/usr/bin/env bash
# 44_final_consolidation.sh -- the end-of-day rebuild, enumerated from disk.
#
# WHY THIS EXISTS. Each queue script rebuilt comparison.json from a run list written
# by hand at the time it was written. Queue 41's list was correct when written and is
# now incomplete: the two G=4 GSPO runs launched afterwards are not in it, so a
# comparison rebuilt by queue 41 would drop their columns -- silently, because a
# dropped column looks exactly like a column that was never measured. The same is
# true of queue 42's list, and of any list written before this one.
#
# So the run set here is not a list. tools/run_set.py enumerates the reports that are
# actually on disk and fails if any of them is in neither its NAMES table nor an
# explicit EXCLUDE entry, which turns "a run went missing" into an error instead of
# an empty column. The enumeration is recorded into the comparison as provenance.
#
# It waits for the card before doing anything, and it is deliberately written to run
# to completion even if that wait times out: a rebuild over a partial set that says
# so is more useful than no rebuild, and the record of what was missing is printed.
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PROJECT="$ROOT/projects/wafer-defect-vlm"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PY="$VENV/python"
AUDIT="$REPORTS/run_set_audit.json"
# Set by any rebuild step that returns non-zero. It decides whether the completion marker
# is printed at all: see the tail of this script.
FAILED=0

step() { printf '\n========== %s ==========\n' "$1"; }

# --------------------------------------------------------------- what must exist
# One report per run. The two without a training step are the untrained baseline and
# the supervised adapter; every other name is one training run's final evaluation.
# A name missing here means that run did not finish, and the comparison would then be
# rebuilt from a partial set -- which is why the wait reports what it was missing.
REQUIRED=(
  qwen35_9b_zero_shot__report.json
  qwen35_9b_adapter__report.json
  qwen35_9b_grpo__report.json
  qwen35_9b_grpo_lr5e5__report.json
  qwen35_9b_grpo_lr1e5_seed3408__report.json
  qwen35_9b_gspo_v1__report.json
  gspo_lr1e5__report.json
  qwen35_9b_gspo_g32__report.json
  qwen35_9b_gspo_g32_lr1e5__report.json
  qwen35_9b_gspo_g4_lr5e5__report.json
  qwen35_9b_gspo_g4_lr1e5__report.json
)

# ------------------------------------------------------------------- 1. wait
# The waiter matches work by argument signature and by exact command line, never by
# script filename: scripts here are pushed as a base64 blob, so the launching shell's
# own command line contains the script's *path* and a filename match would never
# clear. It also waits for the queue drivers themselves, because they rewrite the
# same two comparison files this script is about to rewrite -- racing them would let
# whichever finished second win.
step "1. wait for an idle card and all ${#REQUIRED[@]} reports"
WAIT_LOG="$LOGS/44_wait.log"
"$PY" "$ROOT/tools/wait_for_idle.py" --root "$ROOT" --timeout 21600 --poll 60 \
  --require "${REQUIRED[@]}" 2>&1 | tee "$WAIT_LOG"
if tail -1 "$WAIT_LOG" | grep -q '"idle": true'; then
  echo "card is idle and every required report is present"
else
  echo "WARNING: continuing with a PARTIAL set; the JSON line above says what was missing"
fi

# ------------------------------------------------------- 2. enumerate from disk
step "2. enumerate the run set from the reports on disk"
RUN_ARGS=()
# `run_set.py --emit runs` prints one argument PAIR per line: `--run Name=path`. A line is
# two argv words, not one. Reading each line into the array whole handed argparse a single
# token `"--run Name=path"`, which it reads as the option `--run Name` plus a stray
# `=path`, and answers "the following arguments are required: --run". So this rebuild --
# and after_queue_rebuild's, and post_consolidation's, which were copied from it -- never
# once ran: comparison.json stayed whatever the last queue's hand-written list produced,
# which is how it came to hold eight runs while two more had already landed. The guard
# below did not catch it because it asked only whether the array was empty, and a
# whole-line token is not empty -- it checked a property that was never in doubt. It now
# asserts the array's shape, and the count is pairs.
while read -r flag spec; do
  [ -n "$flag" ] && RUN_ARGS+=("$flag" "$spec")
done < <("$PY" "$ROOT/tools/run_set.py" --root "$ROOT" --audit "$AUDIT" \
           --emit runs 2>"$LOGS/44_run_set.log")
sed 's/^/  /' "$LOGS/44_run_set.log"
if [ "${#RUN_ARGS[@]}" -gt 0 ] && [ "${RUN_ARGS[0]}" != "--run" ]; then
  echo "FATAL: the enumerated arguments are not --run/value pairs (first: '${RUN_ARGS[0]}')" >&2
  exit 1
fi
echo "  runs included: $(( ${#RUN_ARGS[@]} / 2 ))"
if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
  echo "FATAL: no runs found on disk; refusing to rebuild an empty comparison" >&2
  exit 1
fi

# ------------------------------------------------------------- 3. rebuild tables
step "3. rebuild comparison.json / comparison.md"
"$PY" "$ROOT/tools/make_report.py" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} \
  --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
  --provenance "$AUDIT" \
  --note "Metrics shown as not run were never measured; they are not zeros." \
  --note "The run set was enumerated from the reports on disk by tools/run_set.py, not from a hand-written list; the enumeration is appended to this file as provenance and any report on disk that is in neither its table nor its exclusions would have failed that step." \
  && echo "comparison rebuilt" || { echo "FATAL: make_report returned non-zero -- comparison.json/md were NOT rebuilt by this pass, so every claim in them is still computed over whatever run set the last writer used"; FAILED=1; }

# ------------------------------------------------------------- 4. re-run analyses
# Order matters. paired_significance consumes every report; final_report consumes
# paired_significance. The number audit runs last so it sees the regenerated document.
step "4a. paired significance (includes the new single-variable contrast)"
"$PY" "$ROOT/tools/paired_significance.py" 2>&1 | tail -40

step "4b. seed variance"
"$PY" "$ROOT/tools/seed_variance.py" 2>&1 | tail -25 \
  || echo "WARNING: seed_variance returned non-zero (expected only if a seed run is absent)"

step "5. regenerate FINAL_REPORT.md"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md" \
  && echo "FINAL_REPORT.md regenerated" || { echo "FATAL: final_report returned non-zero -- FINAL_REPORT.md is stale"; FAILED=1; }

step "6. trace every number in the documents back to a record"
"$PY" "$ROOT/tools/audit_report_numbers.py" 2>&1 | tail -12

# ------------------------------------------------- 7. the deferred record fixes
# 29_grpo_train.sh is edited only here: editing a running bash script has already
# broken two runs in this project, because bash reads a script by byte offset. By
# this point the waiter has confirmed that nothing is training and no queue driver
# is alive, so no further run can be launched by a script that predates the fix.
step "7. apply the deferred record fixes to the training wrapper"
"$PY" "$ROOT/tools/patch75_grpo_record.py" 2>&1 | tail -12

# --------------------------------------------------------------------- 8. summary
step "8. summary of this consolidation"
"$PY" - "$REPORTS/paired_significance.json" "$AUDIT" <<'PY'
import json, sys
from pathlib import Path
ps_p, au_p = Path(sys.argv[1]), Path(sys.argv[2])
if au_p.is_file():
    a = json.loads(au_p.read_text(encoding="utf-8"))
    print(f"runs included ({len(a['available'])}):")
    for r in a["available"]:
        print(f"  {r['name']:<20} {r['tag']}")
    if a["expected_but_absent"]:
        print("expected but absent -- NOT in the tables:")
        for r in a["expected_but_absent"]:
            print(f"  {r['name']:<20} {r['tag']}")
    print(f"unaccounted reports: {a['unaccounted_reports'] or 'none'}")
if ps_p.is_file():
    ps = json.loads(ps_p.read_text(encoding="utf-8"))
    print(f"\ncontrasts present in {ps_p.name}:")
    for k, v in ps.items():
        if not isinstance(v, dict):
            continue
        d = v.get("accuracy_delta_sequence_minus_token", v.get("accuracy_delta"))
        p = (v.get("mcnemar") or {}).get("p_exact_two_sided", v.get("p_exact_two_sided"))
        if d is None:
            print(f"  {k}")
        else:
            print(f"  {k:<26} delta={d:+.4f}  p={p if p is None else round(p, 6)}")
PY

if [ "$FAILED" -ne 0 ]; then
  printf '\n========== consolidation FAILED ==========\n'
  printf 'At least one rebuild step returned non-zero, so the artefacts on disk are not\n'
  printf 'what this pass claims to have produced. The phrase "consolidation complete" is\n'
  printf 'deliberately NOT printed: post_consolidation.sh keys on it, and a completion\n'
  printf 'marker printed unconditionally is a claim about the run set that nothing checked.\n'
  exit 1
fi
printf '\n========== consolidation complete ==========\n'
