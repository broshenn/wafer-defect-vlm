"""The repair path hands `swift export` a directory that is not an adapter.

`tools/watch_retrieval.sh` is the fallback that fills a report's `retrieval` field when
30_eval_grpo.sh's own step 3b failed -- it is what `post_consolidation.sh` calls, and what
was called by hand tonight when the third seed's merge died with ENOSPC. Its first stage
does:

    swift export --model .../Qwen3.5-9B --adapters "$ROOT/outputs/checkpoints/${TAG}" ...

and training writes its adapters under a run-id and a checkpoint, not at the checkpoint
root:

    outputs/checkpoints/<tag>/v0-<date>-<time>/checkpoint-150/adapter_config.json

so the export answers `AssertionError: ... is not an adapter, please try using --model to
pass it`, the watcher prints `merge FAILED`, and the field stays null. **The merge stage of
this watcher has never worked for any run of this project** -- the earlier runs have the
same layout. The one run whose retrieval was measured (the second seed) was measured by
30_eval_grpo.sh's 3b, which resolves the adapter with a `find` and so was never affected.

The fix is that resolution, copied from 30_eval_grpo.sh: `find -name adapter_config.json`,
`sort -V`, take the last -- newest checkpoint, and `-printf '%h\n'` because the directory
holding the file is the adapter. The comment there explains why it is a `find` and not a
shell loop (a `set -e` loop whose last test fails aborts silently, which once hid a launch
behind an empty log), and that reasoning applies here unchanged.

Two edits: the resolution, and a guard in the merge stage so a missing adapter ends the
attempt loop with the honest outcome (UNMEASURED, which the table renders as "not measured"
in words) instead of calling `swift export` with an empty argument and retrying for half an
hour. The script is parsed with `bash -n` before it is left in place.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
WATCH = ROOT / "tools/watch_retrieval.sh"

OLD_ADAPTER = '  ADAPTER="$ROOT/outputs/checkpoints/${TAG}"\n'
NEW_ADAPTER = '''  # Not the checkpoint root: training writes <root>/<run-id>/checkpoint-N/adapter_config.json,
  # and `swift export` rejects the root with "is not an adapter". Same resolution as
  # 30_eval_grpo.sh, for the reason recorded there -- a `find`, not a shell loop, because
  # under `set -e` a loop whose last iteration fails its test aborts the script silently.
  ADAPTER=$(find "$ROOT/outputs/checkpoints/${TAG}" -maxdepth 3 -name adapter_config.json \\
            -printf '%h\\n' 2>/dev/null | sort -V | tail -1)
  [ -n "$ADAPTER" ] || say "$TAG: WARNING: no adapter_config.json under \\
$ROOT/outputs/checkpoints/${TAG}; the merge cannot run for this run"
'''

OLD_MERGE = '''    if [ ! -s "$RANK" ]; then
      if ! ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
'''
NEW_MERGE = '''    # An empty adapter is not something to retry: without it there is nothing to merge,
    # and the loop below would call `swift export --adapters ""` once a minute for half an
    # hour before saying UNMEASURED. Say it now, and say the honest outcome.
    if [ ! -s "$RANK" ] && [ -z "$ADAPTER" ]; then
      say "$TAG: attempt $attempt: nothing to merge; retrieval stays unmeasured"
      break
    fi
    if [ ! -s "$RANK" ]; then
      if ! ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
'''

s = WATCH.read_text(encoding="utf-8")
if 'ADAPTER=$(find "$ROOT/outputs/checkpoints' in s:
    sys.exit("watch_retrieval.sh already resolves the adapter; nothing written")
for what, old in (("the adapter variable", OLD_ADAPTER), ("the merge stage", OLD_MERGE)):
    if s.count(old) != 1:
        sys.exit(f"{what} matches {s.count(old)} times; nothing written")

shutil.copy2(WATCH, "/tmp/watch_retrieval.sh.bak-patch155")
s = s.replace(OLD_ADAPTER, NEW_ADAPTER, 1).replace(OLD_MERGE, NEW_MERGE, 1)
WATCH.write_text(s, encoding="utf-8")
r = subprocess.run(["bash", "-n", str(WATCH)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/watch_retrieval.sh.bak-patch155", WATCH)
    sys.exit(f"the patched watcher does not parse, restored:\n{r.stderr}")

# The resolution is exercised, not asserted: run the same `find` the script now runs,
# for the tag that failed, and compare with what the eval script resolved for the run
# whose retrieval did get measured.
probe = (
    'for tag in qwen35_9b_grpo_lr1e5_seed3409 qwen35_9b_grpo_lr1e5_seed3408; do '
    'a=$(find /root/autodl-fs/wafer-vlm/outputs/checkpoints/$tag -maxdepth 3 '
    '-name adapter_config.json -printf "%h\\n" 2>/dev/null | sort -V | tail -1); '
    'echo "  $tag -> ${a:-NOTHING}"; done')
p = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
print("the adapter the watcher now resolves:")
print(p.stdout.rstrip())
for line in p.stdout.splitlines():
    if "NOTHING" in line:
        sys.exit("a tag still resolves to no adapter; the watcher would not merge")
print("\nwatch_retrieval.sh parses, and both tags resolve to a real adapter directory")
