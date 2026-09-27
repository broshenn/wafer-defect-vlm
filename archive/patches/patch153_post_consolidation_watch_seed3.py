"""The retrieval-repair list stops one run short of the run it has to repair.

`tools/post_consolidation.sh` holds `WATCH` -- the runs whose evaluation is the last thing
to happen today, with the merged directory each one's eval was told to use. If any of them
ends with `retrieval: null` (its eval's step 3b did not happen), `tools/watch_retrieval.sh`
merges, ranks and re-scores it, or the log ends UNMEASURED so the document can say "not
measured" in words. The list has three entries and the third seed is not one of them: the
seed was launched after this file was written.

What that leaves uncovered is narrow but entirely silent: if the third seed's own 3b fails,
its report keeps a null retrieval, nothing in this pass repairs it, and nothing here notices
that it was not repaired -- the pass prints its completion marker either way. The one loud
thing is `land_run.sh`, which waits six hours for that field and then stops; a six-hour wait
that ends in a stop reads, from outside, exactly like a run that is still training.

This is the fourth hand-written list to be one entry short in one day, and the second one
tonight (see section 8 items 15 and 18): 44's `REQUIRED` was fixed this afternoon by
patch143 with a guard that makes an omission from `run_set.py` fail the step. This list
cannot be made enumerative the same way -- it is not "every run", it is "the runs whose
evaluation happened last", and repairing a baseline report's retrieval is not the same
operation -- so the guard here is the weaker, honest one: the tag and the merged directory
are checked to be the exact strings the run's own launcher uses, and the edited file is
parsed by `bash -n` before it is left in place.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
POST = ROOT / "tools/post_consolidation.sh"
QUEUE = ROOT / "projects/wafer-defect-vlm/scripts/43_queue_grpo_seed3.sh"

OLD = '''WATCH=(
  "qwen35_9b_gspo_g4_lr1e5:Qwen3.5-9B-gspo-g4-lr1e5-merged"
  "qwen35_9b_gspo_g32_lr1e5:Qwen3.5-9B-gspo-g32-lr1e5-merged"
  "qwen35_9b_grpo_lr1e5_seed3408:Qwen3.5-9B-grpo-lr1e5-seed3408-merged"
)
'''
NEW = '''WATCH=(
  "qwen35_9b_gspo_g4_lr1e5:Qwen3.5-9B-gspo-g4-lr1e5-merged"
  "qwen35_9b_gspo_g32_lr1e5:Qwen3.5-9B-gspo-g32-lr1e5-merged"
  "qwen35_9b_grpo_lr1e5_seed3408:Qwen3.5-9B-grpo-lr1e5-seed3408-merged"
  # Added 2026-09-16 18:00. The third seed was launched after this list was written, and
  # a report whose retrieval stays null is a blank cell that reads exactly like a
  # measurement. The tag and the directory are the two strings its own queue script
  # passes to 30_eval_grpo.sh, copied from there rather than re-derived.
  "qwen35_9b_grpo_lr1e5_seed3409:Qwen3.5-9B-grpo-lr1e5-seed3409-merged"
)
'''

s = POST.read_text(encoding="utf-8")
if "seed3409" in s:
    sys.exit("post_consolidation.sh already watches the third seed; nothing written")
if s.count(OLD) != 1:
    sys.exit(f"the WATCH list matches {s.count(OLD)} times; nothing written")

# The two strings, checked against the launcher that produced the run rather than typed.
q = QUEUE.read_text(encoding="utf-8")
TAG = "qwen35_9b_grpo_lr1e5_seed3409"
MERGED = "Qwen3.5-9B-grpo-lr1e5-seed3409-merged"
for what, needle in (("the tag", TAG), ("the merged directory", MERGED)):
    if q.count(needle) < 1:
        sys.exit(f"{what} {needle!r} does not appear in 43_queue_grpo_seed3.sh; the "
                 f"queue would score a different model than this list claims -- "
                 f"nothing written")

shutil.copy2(POST, "/tmp/post_consolidation.sh.bak-patch153")
POST.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run(["bash", "-n", str(POST)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/post_consolidation.sh.bak-patch153", POST)
    sys.exit(f"the patched post_consolidation.sh does not parse, restored:\n{r.stderr}")

p = subprocess.run(["bash", "-c",
                    f'source /dev/stdin <<< "$(sed -n \'/^WATCH=(/,/^)/p\' {POST})"; '
                    f'printf "%s\\n" "${{WATCH[@]}}"'],
                   capture_output=True, text=True)
print("the retrieval-repair list now holds:")
for line in p.stdout.strip().splitlines():
    print("  " + line)
print("\npost_consolidation.sh parses, and the new entry is the tag and merged directory "
      "the seed's own queue script uses")
