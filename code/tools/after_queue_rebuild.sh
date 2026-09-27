#!/usr/bin/env bash
# Rebuild comparison.json / comparison.md / FINAL_REPORT.md from the run set on disk,
# after each of today's two queue scripts exits.
#
# The queue scripts rebuild the comparison from a *hand-written* list of runs, written
# when each script was authored. 41's list has eight entries and names no G=4 run, so
# when it runs its step 5 it replaces a comparison that does contain run 43 with one
# that contains run 41 instead -- and drops run 45, which landed today and is in no
# queue's list at all. 42's list is the same shape with the second seed added. Only
# 44_final_consolidation.sh enumerates the run set from disk, and it waits for run 42,
# which is a couple of hours out.
#
# An omission is not a false number, but it is not harmless either: final_report.py
# computes its claims over the runs it is given, so a sentence like "the highest of all
# runs" is computed over a subset while the sentence says all. That is the failure this
# project records in section 8 item 9, and it would sit in FINAL_REPORT.md -- which no
# checker reads -- until 44 runs.
#
# This does not touch the running scripts. bash reads a script by byte offset as it
# executes (the project found that bug the hard way), so editing a live one is how the
# run gets corrupted. It waits for each to exit and then does what 44 will later do:
# enumerate with tools/run_set.py, rebuild, regenerate.
#
# usage: after_queue_rebuild.sh PID [PID ...]
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"
REPORTS="$ROOT/outputs/reports"
AUDIT="$REPORTS/run_set_audit.json"
LOG="$ROOT/logs/after_queue_rebuild.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

rebuild() {
  local why="$1"
  log "rebuilding the comparison from the reports on disk ($why)"

  local RUN_ARGS=()
  # `run_set.py --emit runs` prints one argument PAIR per line: `--run Name=path`. A line
  # is two argv words, not one, and `mapfile` splits on newlines -- so this handed
  # argparse a single token `"--run Name=path"`, read as the option `--run Name` plus a
  # stray `=path`, and it answered "the following arguments are required: --run". This
  # tool and the two drivers copied from it therefore never rebuilt the comparison; the
  # count below still passed, because a whole-line token does begin with `--run`. The
  # array's shape is asserted now and the count is pairs.
  while read -r flag spec; do
    [ -n "$flag" ] && RUN_ARGS+=("$flag" "$spec")
  done < <("$PY" "$ROOT/tools/run_set.py" --root "$ROOT" --audit "$AUDIT" \
             --emit runs 2>"$ROOT/logs/run_set_after_queue.log")
  if [ "${#RUN_ARGS[@]}" -gt 0 ] && [ "${RUN_ARGS[0]}" != "--run" ]; then
    log "  refusing: enumerated arguments are not --run/value pairs (first: '${RUN_ARGS[0]}')"
    return 1
  fi
  local n=$(( ${#RUN_ARGS[@]} / 2 ))
  log "  run set on disk: $n run(s)"
  if [ "$n" -lt 2 ]; then
    log "  refusing: only $n run(s) enumerated; not overwriting the comparison with that"
    return 1
  fi

  "$PY" "$ROOT/tools/make_report.py" "${RUN_ARGS[@]}" \
    --output "$REPORTS/comparison.json" --markdown "$REPORTS/comparison.md" \
    --provenance "$AUDIT" \
    --note "Metrics shown as not run were never measured; they are not zeros." \
    --note "The run set was enumerated from the reports on disk by tools/run_set.py, not from a hand-written list; the enumeration is appended to this file as provenance." \
    && log "  comparison rebuilt ($n runs)" \
    || { log "  FATAL: make_report.py returned non-zero -- the comparison was NOT rebuilt, so the copy on disk still describes whatever run set its last successful writer used"; return 1; }

  "$PY" "$ROOT/tools/final_report.py" --root "$ROOT" --output "$ROOT/FINAL_REPORT.md" \
    && log "  FINAL_REPORT.md regenerated" \
    || log "  WARNING: final_report.py returned non-zero"
}

for pid in "$@"; do
  if kill -0 "$pid" 2>/dev/null; then
    log "waiting for pid $pid to exit"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    log "pid $pid has exited"
    sleep 20   # let the last writes settle before reading the reports
    rebuild "after pid $pid"
  else
    log "pid $pid is not running; nothing to wait for"
  fi
done
log "done"
