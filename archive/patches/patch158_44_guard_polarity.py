"""44's wait-list guard fails exactly when it passes.

`44_final_consolidation.sh` line 60 asks `tools/run_set.py` what the run set is and
refuses to continue if the script's own `REQUIRED` wait list is one run short (patch143
added it this afternoon, after the third seed was missing from that list). Tonight it
printed, in this order:

    the wait list covers every run run_set.py accounts for (12)
    FATAL: the wait list does not cover the enumerated run set (see above)

Both lines are true statements about different things: the first is the guard's own
success message, the second is the shell's `else` branch. The construct is

    if ! "$PY" - "$ROOT" "${REQUIRED[@]}" <<'GUARDPY'
    ...
    GUARDPY
    then
      echo "wait list check: covers the enumerated run set"
    else
      echo "FATAL: the wait list does not cover the enumerated run set (see above)"
      FAILED=1
    fi

so the success branch runs when the command *fails*, and `FAILED=1` -- which withholds
the completion marker -- is set when the command *succeeds*. The two branches are
swapped relative to their messages.

The consequence is not a wrong number, it is a run that can never finish: 44's file was
last written at 17:34, the phrase "consolidation complete" appears nowhere in the logs
of any run since, and `post_consolidation.sh` keys its first step on that marker.
Nobody saw it because this was the guard's first execution -- patch143 wrote it at
17:34 and the next time 44 ran was tonight at 18:57.

The fix is to drop the `!`: the messages then sit under the outcomes they describe. The
polarity is exercised in both directions on the patched text before the file is left in
place, with `true` and `false` standing in for the check -- the same thing patch75's
`bash -n` gate taught tonight, that a check which has never been run is a line of prose.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
F = ROOT / "projects/wafer-defect-vlm/scripts/44_final_consolidation.sh"

OLD = "if ! \"$PY\" - \"$ROOT\" \"${REQUIRED[@]}\" <<'GUARDPY'"
NEW = """# The polarity matters and was wrong until 2026-09-16: with `if !`, this guard printed
# its FATAL and withheld the completion marker exactly when the wait list was complete.
# It was written at 17:34 and first executed at 18:57, which is why nothing had noticed.
# Both directions are exercised below the fix, in the patch that made it.
if "$PY" - "$ROOT" "${REQUIRED[@]}" <<'GUARDPY'"""

MARK = "The polarity matters and was wrong until 2026-09-16"

s = F.read_text(encoding="utf-8")
if MARK in s:
    sys.exit("the guard already has the fixed polarity; nothing written")
if s.count(OLD) != 1:
    sys.exit(f"the guard's first line matches {s.count(OLD)} times; nothing written")

shutil.copy2(F, "/tmp/44_final_consolidation.sh.bak-patch158")
s2 = s.replace(OLD, NEW, 1)
F.write_text(s2, encoding="utf-8")

r = subprocess.run(["bash", "-n", str(F)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/44_final_consolidation.sh.bak-patch158", F)
    sys.exit(f"the patched script does not parse, restored:\n{r.stderr}")

BLOCK_END = """else
  echo "FATAL: the wait list does not cover the enumerated run set (see above)"
  FAILED=1
fi"""


def guard_script(text, check_exit):
    """The guard's real tail -- from the heredoc to its closing `fi` -- with the check
    itself replaced by `true` or `false`, so only the shell's polarity is under test.

    The `!` is read out of the version under test rather than written here: the first
    draft of this probe hardcoded `if true`, which tested the same polarity twice and
    reported that the unpatched guard was already correct."""
    i = text.index("<<'GUARDPY'")
    line_start = text.rindex("\n", 0, i) + 1
    if_line = text[line_start:i]
    negation = "! " if "!" in if_line else ""
    body_start = text.index("\n", i) + 1
    body_end = text.index("\nGUARDPY", body_start)
    j = text.index(BLOCK_END, body_end) + len(BLOCK_END)
    tail = text[body_end:j].lstrip("\n").split("\n", 1)[1]
    head = "if " + negation + ("true" if check_exit == 0 else "false") + "\n"
    return "FAILED=0\n" + head + tail + "\nprintf 'FAILED=%s\\n' \"$FAILED\"\n"


def probe(text, check_exit):
    p = subprocess.run(["bash", "-c", guard_script(text, check_exit)],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    branch = ("covers" if "wait list check: covers" in out
              else "FATAL" if "FATAL" in out else "NEITHER")
    return branch, ("FAILED=1" in out)


old = pathlib.Path("/tmp/44_final_consolidation.sh.bak-patch158").read_text(encoding="utf-8")
before = {rc: probe(old, rc) for rc in (0, 1)}
after = {rc: probe(s2, rc) for rc in (0, 1)}
for name, res in (("before", before), ("after", after)):
    for rc in (0, 1):
        branch, failed = res[rc]
        print(f"  {name:6s} check exits {rc}: branch={branch:6s} FAILED=1={failed}")

# Before: exit 0 (the list is complete) must reach the FATAL branch and set FAILED.
# After: exit 0 must reach the success branch and leave FAILED alone; exit 1 must do the
# opposite. If any of the four readings is wrong, the patch is not what it claims.
if not (before[0] == ("FATAL", True) and before[1] == ("covers", False)):
    shutil.copy2("/tmp/44_final_consolidation.sh.bak-patch158", F)
    sys.exit(f"the unpatched guard did not read as inverted ({before}); restored")
if not (after[0] == ("covers", False) and after[1] == ("FATAL", True)):
    shutil.copy2("/tmp/44_final_consolidation.sh.bak-patch158", F)
    sys.exit(f"the patched guard does not separate the two outcomes ({after}); restored")

print("\n44_final_consolidation.sh: the guard's branches are the right way round, "
      "verified in both directions")
