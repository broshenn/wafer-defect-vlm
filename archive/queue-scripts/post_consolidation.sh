#!/usr/bin/env bash
# After 44 has finished: make sure every report actually carries its retrieval
# numbers, then re-derive everything downstream so the tables describe the reports
# that are on disk *now*.
#
# WHY THIS HAS TO RUN AFTER 44, NOT ALONGSIDE IT. scripts/30_eval_grpo.sh fills the
# report's `retrieval` field itself (step 3b: merge the adapter, rank the benchmark,
# re-score). A report that is still `retrieval: null` when 44 runs is therefore a
# report whose 3b did not happen -- and a null there renders in 5.2.2 as a blank
# cell, which is indistinguishable from "measured, and it was zero". Repairing that
# while 44 is reading the same files would race it: 44's waiter treats "no training,
# no eval, no queue driver" as idle, and a repair pass is none of those. So the
# repair is sequenced after 44's completion marker and then 44's own rebuild chain is
# re-run, rather than being interleaved with it.
#
# The repair never invents a number: either a real merge+rank+re-score fills the
# field, or the log ends UNMEASURED so the document can say "not measured" in words.
#
# Idempotent: if every report already has retrieval, the repair exits immediately and
# the re-derivation reproduces byte-identical tables (that is what makes it safe to
# run it a second time by hand).
set -uo pipefail

ROOT="${WAFER_ROOT:-/root/autodl-fs/wafer-vlm}"
VENV="$ROOT/venvs/wafer/bin"
PY="$VENV/python"
REPORTS="$ROOT/outputs/reports"
LOGS="$ROOT/logs"
PROJECT="$ROOT/projects/wafer-defect-vlm"
AUDIT="$REPORTS/run_set_audit.json"
# Set by any re-derivation step that returns non-zero; decides the completion marker
# at the tail of this script (see 44_final_consolidation.sh for the same rule).
FAILED=0
CONSOLIDATION_LOG="$LOGS/44_driver.log"

step() { printf '\n========== %s ==========\n' "$1"; }

# The three runs whose evaluation is the last thing to happen today, with the merged
# directory each one's eval was told to use. A wrong directory here would score a
# different model than the run's own column claims, so these are the same strings
# 30_eval_grpo.sh is called with by that run's driver, not re-derived.
WATCH=(
  "qwen35_9b_gspo_g4_lr1e5:Qwen3.5-9B-gspo-g4-lr1e5-merged"
  "qwen35_9b_gspo_g32_lr1e5:Qwen3.5-9B-gspo-g32-lr1e5-merged"
  "qwen35_9b_grpo_lr1e5_seed3408:Qwen3.5-9B-grpo-lr1e5-seed3408-merged"
)

step "1. wait for 44_final_consolidation.sh to finish"
waited=0
until grep -q "consolidation complete" "$CONSOLIDATION_LOG" 2>/dev/null; do
  if ! pgrep -f "44_final_consolidation.sh" >/dev/null 2>&1; then
    echo "WARNING: 44 is gone without its completion marker; continuing anyway and" \
         "recording that this pass ran without a confirmed predecessor"
    break
  fi
  [ "$waited" -ge 28800 ] && { echo "WARNING: 44 still running after ${waited}s;" \
    "continuing anyway" ; break; }
  [ $((waited % 600)) -eq 0 ] && echo "  waiting on 44 (${waited}s)"
  sleep 60
  waited=$((waited + 60))
done
echo "44 finished after ~${waited}s (or was already done)"

# ------------------------------------------------------------ 2. fingerprint
# What "changed" means here is the report's own retrieval field, not its mtime: a
# re-score writes the same file whether or not it measured anything, so an mtime test
# would report a change every time and hide a real one behind it.
step "2. retrieval state before the repair"
"$PY" - "${WATCH[@]}" <<'PY'
import json, sys
from pathlib import Path
rep = Path("/root/autodl-fs/wafer-vlm/outputs/reports")
for spec in sys.argv[1:]:
    tag = spec.split(":")[0]
    p = rep / f"{tag}__report.json"
    if not p.is_file():
        print(f"  {tag:<32} no report on disk"); continue
    r = json.loads(p.read_text(encoding="utf-8")).get("retrieval")
    print(f"  {tag:<32} " + ("retrieval: NULL (must be repaired or declared "
          "unmeasured)" if r is None else f"retrieval present, mAP@10={r.get('mAP@10')}"))
PY

step "3. repair any report whose retrieval is still null"
bash "$ROOT/tools/watch_retrieval.sh" "${WATCH[@]}" 2>&1 | tee "$LOGS/post_retrieval_repair.log"

# ------------------------------------------------------------ 4. re-derive
# Same chain as 44's steps 2-6, run again so the tables are built from the reports as
# they are after the repair. run_set.py enumerates the reports on disk; nothing here
# is a hand-written run list.
step "4a. enumerate the run set from the reports on disk"
RUN_ARGS=()
# One line of `run_set.py --emit runs` is an argument PAIR (`--run Name=path`), i.e. two
# argv words. Pushing the line whole gave argparse a single token `"--run Name=path"` --
# read as the option `--run Name` plus a stray `=path` -- so it answered "the following
# arguments are required: --run" and this step, like the two others copied from it, never
# rebuilt anything. The shape is asserted below now, not just the array's non-emptiness.
while read -r flag spec; do
  [ -n "$flag" ] && RUN_ARGS+=("$flag" "$spec")
done < <("$PY" "$ROOT/tools/run_set.py" --root "$ROOT" --audit "$AUDIT" \
           --emit runs 2>"$LOGS/post_run_set.log")
sed 's/^/  /' "$LOGS/post_run_set.log"
if [ "${#RUN_ARGS[@]}" -gt 0 ] && [ "${RUN_ARGS[0]}" != "--run" ]; then
  echo "FATAL: the enumerated arguments are not --run/value pairs (first: '${RUN_ARGS[0]}')" >&2
  exit 1
fi
echo "  runs included: $(( ${#RUN_ARGS[@]} / 2 ))"
if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
  echo "FATAL: no runs found on disk; refusing to rebuild an empty comparison" >&2
  exit 1
fi

step "4b. rebuild comparison.json / comparison.md"
"$PY" "$ROOT/tools/make_report.py" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} \
  --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
  --provenance "$AUDIT" \
  --note "Metrics shown as not run were never measured; they are not zeros." \
  --note "The run set was enumerated from the reports on disk by tools/run_set.py, not from a hand-written list; the enumeration is appended to this file as provenance and any report on disk that is in neither its table nor its exclusions would have failed that step." \
  && echo "comparison rebuilt" || { echo "FATAL: make_report returned non-zero -- comparison.json/md were NOT rebuilt by this pass"; FAILED=1; }

step "4c. paired significance"
"$PY" "$ROOT/tools/paired_significance.py" 2>&1 | tail -40

step "4d. seed variance"
"$PY" "$ROOT/tools/seed_variance.py" 2>&1 | tail -25 \
  || echo "WARNING: seed_variance returned non-zero (expected only if a seed run is absent)"

step "4e. regenerate FINAL_REPORT.md"
"$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md" \
  && echo "FINAL_REPORT.md regenerated" || { echo "FATAL: final_report returned non-zero -- FINAL_REPORT.md is stale"; FAILED=1; }

# --------------------------------------------------- 5. counts, then the audit
# sync_counts rewrites LIMITATIONS.md's "how many runs" sentences from the records, so
# it must run before the audit: a count that is one run stale is exactly what the
# audit is there to catch, and running them in the other order would have the audit
# bless the stale one.
step "5a. re-sync the document's run counts from the records"
"$PY" "$ROOT/tools/sync_counts.py" 2>&1 | tail -20 \
  || echo "WARNING: sync_counts returned non-zero; the counts in LIMITATIONS.md may lag"

step "5b. trace every number in the documents back to a record"
"$PY" "$ROOT/tools/audit_report_numbers.py" 2>&1 | tail -12

step "6. final state"
"$PY" - "${WATCH[@]}" <<'PY'
import json, sys
from pathlib import Path
rep = Path("/root/autodl-fs/wafer-vlm/outputs/reports")
bad = []
for spec in sys.argv[1:]:
    tag = spec.split(":")[0]
    p = rep / f"{tag}__report.json"
    r = json.loads(p.read_text(encoding="utf-8")).get("retrieval") if p.is_file() else None
    if r is None:
        bad.append(tag)
    else:
        print(f"  {tag:<32} mAP@10={r.get('mAP@10')}")
print("UNMEASURED: " + ", ".join(bad) if bad
      else "every watched report carries its retrieval numbers")
PY

if [ "$FAILED" -ne 0 ]; then
  printf '\n========== post-consolidation pass FAILED ==========\n'
  printf 'At least one re-derivation step returned non-zero, so the artefacts on disk\n'
  printf 'are not what this pass claims to have produced.\n'
  exit 1
fi
printf '\n========== post-consolidation pass complete ==========\n'
