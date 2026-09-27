"""Two things tonight's chain exposed about its own completion markers.

FIRST, a sentence that claims a record it does not make. `tools/post_consolidation.sh`
waits for 44_final_consolidation.sh's completion marker by grepping
`logs/44_driver.log`, and when the marker is not there and 44 is not running it prints:

    WARNING: 44 is gone without its completion marker; continuing anyway and
             recording that this pass ran without a confirmed predecessor

Nothing records it. The word "predecessor" occurs exactly once in the whole file, in
that sentence: no flag is set, no artefact is written, and the pass ends with its
ordinary completion marker, which says nothing about the predecessor. So the one state a
reader most needs to know about -- that this pass rebuilt the documents without a
confirmed predecessor -- is announced as recorded and then not recorded. The fix here is
the weak, honest one available: a flag, and a completion marker that carries the
qualifier whenever it is set, so the thing readers key on says what happened. It is not a
new check; it is the removal of a false claim.

SECOND, why that path was taken tonight, which was my own wrapper's fault. The queue
scripts feed 44's output to `logs/44_driver.log`, which is where the marker is looked
for. `tools/finish_chain_seed3409.sh`, written tonight to sequence the three end-of-day
stages, redirected every stage's output into one chain log -- so 44 printed its marker
into a file nobody greps, and post_consolidation took the unconfirmed path. Both facts
are worth keeping: the marker handshake is a path, not a promise, and centralising the
logs broke it without anything failing.

The chain script is corrected to tee 44's output into `logs/44_driver.log` as well,
carrying 44's exit status out of the pipe with PIPESTATUS (a pipe would otherwise report
tee's status and turn a failed consolidation into a successful one).
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
POST = ROOT / "tools/post_consolidation.sh"
CHAIN = ROOT / "tools/finish_chain_seed3409.sh"

# ---- post_consolidation.sh: the unconfirmed-predecessor state becomes real -----------
OLD_WARN = """    echo "WARNING: 44 is gone without its completion marker; continuing anyway and" \\
         "recording that this pass ran without a confirmed predecessor"
    break
"""
NEW_WARN = """    echo "WARNING: 44 is gone without its completion marker; continuing anyway. The" \\
         "completion marker at the end of this pass will carry that fact -- until" \\
         "2026-09-16 this line said it was recording it and recorded it nowhere: the" \\
         "word 'predecessor' appeared once in this file, in this sentence."
    UNCONFIRMED=1
    break
"""

OLD_TIMEOUT = """  [ "$waited" -ge 28800 ] && { echo "WARNING: 44 still running after ${waited}s;" \\
    "continuing anyway" ; break; }
"""
NEW_TIMEOUT = """  [ "$waited" -ge 28800 ] && { echo "WARNING: 44 still running after ${waited}s;" \\
    "continuing anyway" ; UNCONFIRMED=1; break; }
"""

OLD_DECL = """# Set by any re-derivation step that returns non-zero; decides the completion marker
# at the tail of this script (see 44_final_consolidation.sh for the same rule).
FAILED=0
"""
NEW_DECL = """# Set by any re-derivation step that returns non-zero; decides the completion marker
# at the tail of this script (see 44_final_consolidation.sh for the same rule).
FAILED=0
# Set when step 1 gave up waiting: 44 is gone without its marker, or is still running
# after the eight-hour cap. Both mean this pass rebuilt the documents without a
# confirmed predecessor, and the tail marker says so instead of saying nothing.
UNCONFIRMED=0
"""

OLD_MARK = "printf '\\n========== post-consolidation pass complete ==========\\n'\n"
NEW_MARK = """if [ "$UNCONFIRMED" -ne 0 ]; then
  printf '\\n========== post-consolidation pass complete (WITHOUT a confirmed predecessor) ==========\\n'
  printf 'Step 1 did not see 44_final_consolidation.sh'"'"'s completion marker, so the documents\\n'
  printf 'rebuilt above may be derived from a run set 44 had not finished with. This line\\n'
  printf 'is the record of that: until 2026-09-16 the earlier WARNING claimed to record it\\n'
  printf 'and nothing did. Read the two documents, not this marker, for the run set.\\n'
else
  printf '\\n========== post-consolidation pass complete ==========\\n'
fi
"""

# ---- the chain script: 44's marker has to reach the file it is looked for in ---------
OLD_CHAIN = """if ! bash "$ROOT/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh"; then
  say "STOP: 44 failed. post_consolidation is NOT run."
  exit 1
fi
"""
NEW_CHAIN = """# 44's output goes to this chain's log through the redirection above, but
# post_consolidation.sh looks for 44's completion marker in logs/44_driver.log -- the
# file the queue scripts feed it to. When this chain was first run, 44's marker landed
# in the chain log and post_consolidation took its "no confirmed predecessor" path: the
# handshake is a path, and centralising the logs broke it silently. `tee` puts it back
# where it is read, and PIPESTATUS carries 44's own exit status out of the pipe -- with
# a bare pipe, $? would be tee's, and a failed consolidation would look successful.
bash "$ROOT/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh" 2>&1 \\
  | tee -a "$ROOT/logs/44_driver.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  say "STOP: 44 failed (exit $rc). post_consolidation is NOT run."
  exit 1
fi
"""

edits = [
    (POST, "the unconfirmed-predecessor WARNING", OLD_WARN, NEW_WARN),
    (POST, "the eight-hour timeout branch", OLD_TIMEOUT, NEW_TIMEOUT),
    (POST, "the FAILED declaration", OLD_DECL, NEW_DECL),
    (POST, "the completion marker", OLD_MARK, NEW_MARK),
    (CHAIN, "the stage-2 invocation", OLD_CHAIN, NEW_CHAIN),
]

for path, what, old, new in edits:
    s = path.read_text(encoding="utf-8")
    # Test the whole block, not its first line: two of these edits keep the first line
    # and change only what follows, and a first-line test called those already patched.
    if new in s:
        sys.exit(f"{what} in {path.name} is already patched; nothing written")
    if s.count(old) != 1:
        sys.exit(f"{what} in {path.name} matches {s.count(old)} times; nothing written")

for path in (POST, CHAIN):
    shutil.copy2(path, f"/tmp/{path.name}.bak-patch159")

for path, what, old, new in edits:
    s = path.read_text(encoding="utf-8")
    path.write_text(s.replace(old, new, 1), encoding="utf-8")

for path in (POST, CHAIN):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        for p in (POST, CHAIN):
            shutil.copy2(f"/tmp/{p.name}.bak-patch159", p)
        sys.exit(f"{path.name} does not parse after the edits; both files restored:\n"
                 f"{r.stderr}")

# The marker has to actually change with the flag, so both settings are exercised on
# the real tail of the file rather than trusted.
print("the completion marker, read from the patched file for both settings:")
for flag in (0, 1):
    p = subprocess.run(["bash", "-c", f"UNCONFIRMED={flag}\n" + NEW_MARK],
                       capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if "pass complete" in l]
    print(f"  UNCONFIRMED={flag}: {line[0].strip() if line else 'NO MARKER'}")

p1 = subprocess.run(["bash", "-c", "UNCONFIRMED=1\n" + NEW_MARK],
                    capture_output=True, text=True).stdout
p0 = subprocess.run(["bash", "-c", "UNCONFIRMED=0\n" + NEW_MARK],
                    capture_output=True, text=True).stdout
if "WITHOUT a confirmed predecessor" not in p1 or "WITHOUT" in p0:
    for p in (POST, CHAIN):
        shutil.copy2(f"/tmp/{p.name}.bak-patch159", p)
    sys.exit("the marker does not separate the two states; both files restored")

print(f"\npost_consolidation.sh: the marker carries the qualifier, and 'predecessor' now "
      f"appears {POST.read_text(encoding='utf-8').count('predecessor')} times")
print("finish_chain_seed3409.sh: 44's output is teed to logs/44_driver.log where the "
      "marker is looked for")
