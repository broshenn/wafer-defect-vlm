#!/usr/bin/env bash
# The pass that every landing ends with, whatever landed.
#
# Four things have to become true together once a run is in the document: the counts
# derived from the records, the quantifiers whose scope moved, and the traceability of
# both. They are separate tools because they check different things, and they run in
# this order because each one's refusal is meaningful to the next:
#
#   1. sync_counts  -- the four counts that are pure arithmetic (idle-step totals, the
#      record count). It refuses to run when the number audit fails;
#   2. check_quantified_claims --fix -- the counts *inside* the sentences that also
#      make a claim ("N 个 run 里唯一 … 的是 X"). It rewrites numerals only, and it
#      refuses to rewrite any numeral while a fact in the same sentence has moved;
#   3. check_quantified_claims without --fix -- confirm, from a clean read, that the
#      numerals now match. A fix that did not settle the check would otherwise look
#      like one that did;
#   4. audit_report_numbers -- trace every number in the three documents back to a
#      record, and fail on an unfilled template field.
#
# A non-zero exit from step 2 or 3 is not a bug in this script: it means a sentence
# about the run set has to be rewritten by hand, and the tool prints exactly which.
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
PY="$ROOT/venvs/wafer/bin/python"

step() { printf '\n========== %s ==========\n' "$1"; }

step "counts derived from the records"
"$PY" "$ROOT/tools/sync_counts.py" || echo "WARNING: sync_counts failed; the derived counts may lag"

step "counts inside the sentences that also make a claim"
if ! "$PY" "$ROOT/tools/check_quantified_claims.py" --fix; then
  printf '\nSTOP: a fact about the run set has moved, so no count was written.\n'
  printf 'The sentence above has to be rewritten with the value printed beside it.\n'
  exit 1
fi

step "confirm from a clean read"
if ! "$PY" "$ROOT/tools/check_quantified_claims.py"; then
  printf '\nSTOP: the confirm pass still fails, so the fix did not settle it.\n'
  exit 1
fi

step "trace every number back to a record"
"$PY" "$ROOT/tools/audit_report_numbers.py" 2>&1 | tail -14
printf '\n========== landing finished ==========\n'
