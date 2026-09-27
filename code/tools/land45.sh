#!/usr/bin/env bash
# Land run 45 (GSPO, G=4, lr 1e-5) into the documents, in the one order that works.
#
# The order is not arbitrary:
#   1. the report must carry its retrieval numbers before a column is built from it,
#      because a null retrieval renders as a blank cell and a blank cell in this
#      document reads as a measurement that came out small (that is what
#      add_run_columns.py refuses to write without --allow-unmeasured);
#   2. the column goes in before the counts are re-synced, because one of the counts
#      is the number of columns in 5.2.5's per-class table, read from the table;
#   3. paired_significance is re-run before patch92b, because patch92b quotes the
#      G=4 lr pair out of that record and exits if it is not there;
#   4. sync_counts runs after the prose edits, and the audit after sync_counts --
#      a count that is one run stale is exactly what the audit is meant to catch, so
#      running them in the other order would have the audit bless the stale one.
#
# It refuses rather than proceeding on a partial report: if retrieval never appears
# within the window it stops, because a column added without it would look complete.
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"
TAG=qwen35_9b_gspo_g4_lr1e5
REP="$ROOT/outputs/reports/${TAG}__report.json"
LOG="$ROOT/logs/land45.log"

step() { printf '\n========== %s ==========\n' "$1"; }

step "1. wait for $TAG's report to carry retrieval"
waited=0
until [ -s "$REP" ] && "$PY" -c "
import json, sys
sys.exit(0 if json.load(open('$REP', encoding='utf-8')).get('retrieval') else 1)
" 2>/dev/null; do
  if [ "$waited" -ge 14400 ]; then
    echo "STOP: $TAG's retrieval never appeared within ${waited}s. No column is"
    echo "added: without retrieval the cell would be blank, and a blank cell here"
    echo "reads as a measurement that came out small. Report: $REP"
    exit 1
  fi
  [ $((waited % 600)) -eq 0 ] && echo "  waiting (${waited}s)"
  sleep 60
  waited=$((waited + 60))
done
echo "report with retrieval present after ~${waited}s"

step "2. add the column to the four metrics tables"
"$PY" "$ROOT/tools/add_run_columns.py" --tag "$TAG" --name 'GSPO(G=4) lr1e-5' \
  --after 'GSPO(G=4) lr5e-5' || {
    echo "STOP: add_run_columns refused; nothing further is written, because every"
    echo "step below quotes a table this one was supposed to change."
    exit 1
  }

step "3. re-run the paired record (it now has the G=4 lr pair)"
"$PY" "$ROOT/tools/paired_significance.py" > "$ROOT/logs/land45_paired.log" 2>&1 || {
    echo "STOP: paired_significance failed; nothing written:"
    tail -20 "$ROOT/logs/land45_paired.log"
    exit 1
  }
tail -25 "$ROOT/logs/land45_paired.log"

step "4. write the run-45 prose into LIMITATIONS.md"
"$PY" "$ROOT/tools/patch92b_run45.py" || exit 1

step "5. re-sync the document's run counts from the records"
"$PY" "$ROOT/tools/sync_counts.py" || echo "WARNING: sync_counts failed; the counts may lag"

step "6. trace every number in the documents back to a record"
"$PY" "$ROOT/tools/audit_report_numbers.py" 2>&1 | tail -14

printf '\n========== run 45 landed ==========\n'
