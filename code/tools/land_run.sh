#!/usr/bin/env bash
# Land one finished run into the documents, in the one order that works.
#
# Three runs finish today and each one changes the same four things: a column in the
# four metrics tables, the paired record, the prose that quotes them, and the counts
# that count the runs. Doing that by hand three times is how a count ends up one run
# stale -- the failure this project keeps finding, where every number is right and
# the sentence is about a run set that no longer exists. So the sequence is a script.
#
# The order:
#   1. the report must carry its retrieval numbers before a column is built from it:
#      a null retrieval renders as a blank cell, and a blank cell here reads as a
#      measurement that came out small (add_run_columns.py refuses to write one
#      without --allow-unmeasured, and that refusal is the point);
#   2. any --pre tools run before the column, because they consume the report;
#   3. the column goes in before the counts are re-synced, because one of the counts
#      is the number of columns in 5.2.5's per-class table, read from the table;
#   4. paired_significance is re-run before the --patch scripts, because they quote
#      pairs out of that record and exit if the pair is not there;
#   5. sync_counts runs after the prose edits and the audit after sync_counts: a
#      count that is one run stale is exactly what the audit is for, so the other
#      order would have the audit bless the stale one.
#
# It checks that every --patch file exists BEFORE it changes anything. A lander that
# added the column and then died on a missing patch script would leave the document
# half-updated, which is worse than not starting.
#
# usage: land_run.sh --tag T [--name N --after A] [--no-column]
#                    [--pre tool.py]... [--patch p.py]...
#
# --no-column is for a run that cannot be a column. Run 42 is a second seed of an
# existing cell (GRPO, G=4, lr1e-5), and `colkey()` identifies a column by
# (algorithm, group size, learning rate) -- two seeds of one cell are the same column
# by that key, so the table has no place to put it and the seed finding is prose.
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"
TAG="" NAME="" AFTER="" NOCOL=0
PRE=() PATCH=()

step() { printf '\n========== %s ==========\n' "$1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)   TAG="$2";   shift 2 ;;
    --name)  NAME="$2";  shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --no-column) NOCOL=1; shift ;;
    --pre)   PRE+=("$2");   shift 2 ;;
    --patch) PATCH+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TAG" ] || { echo "usage: land_run.sh --tag T [--name N --after A]" >&2; exit 2; }
if [ "$NOCOL" -eq 0 ]; then
  [ -n "$NAME" ] && [ -n "$AFTER" ] || {
    echo "without --no-column, --name and --after are required" >&2; exit 2; }
fi

REP="$ROOT/outputs/reports/${TAG}__report.json"

# --- everything that must exist before anything is written ---------------------
for f in "${PATCH[@]:-}"; do
  [ -n "$f" ] || continue
  [ -s "$ROOT/$f" ] || { echo "STOP: patch script $f is missing or empty; nothing written" >&2; exit 1; }
done

step "$TAG 1. wait for the report to carry retrieval"
waited=0
until [ -s "$REP" ] && "$PY" -c "
import json, sys
sys.exit(0 if json.load(open('$REP', encoding='utf-8')).get('retrieval') else 1)
" 2>/dev/null; do
  if [ "$waited" -ge 21600 ]; then
    echo "STOP: $TAG's retrieval never appeared within ${waited}s. No column is added:"
    echo "without it the cell would be blank, and a blank cell here reads as a"
    echo "measurement that came out small. Report: $REP"
    exit 1
  fi
  [ $((waited % 600)) -eq 0 ] && echo "  waiting (${waited}s)"
  sleep 60
  waited=$((waited + 60))
done
echo "report with retrieval present after ~${waited}s"

for t in "${PRE[@]:-}"; do
  [ -n "$t" ] || continue
  step "$TAG 2. pre-tool $t"
  "$PY" "$ROOT/$t" || { echo "STOP: $t failed; nothing further written" >&2; exit 1; }
done

if [ "$NOCOL" -eq 0 ]; then
  step "$TAG 3. add the column to the four metrics tables"
  "$PY" "$ROOT/tools/add_run_columns.py" --tag "$TAG" --name "$NAME" --after "$AFTER" || {
    echo "STOP: add_run_columns refused; nothing further is written, because every step"
    echo "below quotes a table this one was supposed to change."
    exit 1
  }
else
  step "$TAG 3. no column: this run is a second seed of an existing cell"
fi

step "$TAG 4. re-run the paired record"
"$PY" "$ROOT/tools/paired_significance.py" > "$ROOT/logs/land_${TAG}_paired.log" 2>&1 || {
  echo "STOP: paired_significance failed; nothing written:"; tail -20 "$ROOT/logs/land_${TAG}_paired.log"
  exit 1
}
tail -20 "$ROOT/logs/land_${TAG}_paired.log"

for p in "${PATCH[@]:-}"; do
  [ -n "$p" ] || continue
  step "$TAG 5. prose: $p"
  "$PY" "$ROOT/$p" || { echo "STOP: $p failed; the column is in but the prose is not" >&2; exit 1; }
done

step "$TAG 6. counts, quantifiers, and traceability"
bash "$ROOT/tools/land_finish.sh" || {
  echo "STOP: the finishing pass has something for you -- see above. The column and the"
  echo "paired record are in; a sentence about the run set is not."
  exit 1
}

printf '\n========== %s landed ==========\n' "$TAG"
