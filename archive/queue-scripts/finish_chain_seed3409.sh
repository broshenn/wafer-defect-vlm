#!/usr/bin/env bash
# The end-of-day chain for the third seed, in the order queue 43 intended.
#
# Queue 43's own chain (tools/finish_chain_after_seed3.sh) was killed tonight to break
# the deadlock between the landing and the retrieval repair, so the three stages run
# here instead, sequentially and each only on the previous one's success:
#
#   1. tools/land_finish.sh            -- the finishing pass the landing stopped in.
#      It stopped because section 8 item 11's table was one row short: the third seed
#      has a record and was not in it. That row is in (patch157), so the pass is re-run
#      rather than the landing, which would re-run the --patch scripts and be refused
#      by each of them for already having written.
#   2. scripts/44_final_consolidation.sh -- rebuilds comparison.md, the paired record
#      and FINAL_REPORT.md from the run set enumerated from disk.
#   3. tools/post_consolidation.sh     -- waits for 44's completion marker, repairs any
#      retrieval that is still null, then re-derives the same documents from there.
#
# Stage 2 runs only if stage 1 exits 0. That is not caution for its own sake: 44
# enumerates the run set from disk and rebuilds the documents from whatever it finds,
# so running it on a half-landed state would publish documents derived from a run set
# whose prose has not been rewritten yet -- which is exactly what queue 43's STOP
# branch refuses to do when it says the comparison is deliberately not rebuilt.
set -uo pipefail

ROOT=/root/autodl-fs/wafer-vlm
TAG=qwen35_9b_grpo_lr1e5_seed3409
LOG="$ROOT/logs/finish_chain_seed3409_$(date +%H%M%S).log"

: > "$LOG"
exec >> "$LOG" 2>&1

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "chain started; log is $LOG"

say "stage 1/3: tools/land_finish.sh (the pass the landing stopped in)"
if ! bash "$ROOT/tools/land_finish.sh"; then
  say "STOP: land_finish failed. Stages 2 and 3 are NOT run: 44 rebuilds the documents"
  say "from the run set on disk, and doing that while a sentence about the run set is"
  say "still wrong would publish the same wrongness with a fresh timestamp. The log"
  say "above names the sentence or the tool to fix."
  exit 1
fi
say "stage 1 ok: the finishing pass is clean"

say "stage 2/3: 44_final_consolidation.sh"
if ! bash "$ROOT/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh"; then
  say "STOP: 44 failed. post_consolidation is NOT run."
  exit 1
fi
say "stage 2 ok"

say "stage 3/3: tools/post_consolidation.sh"
if ! bash "$ROOT/tools/post_consolidation.sh"; then
  say "STOP: post_consolidation failed -- see above."
  exit 1
fi
say "stage 3 ok"
say "chain complete for $TAG"
