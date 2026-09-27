"""Fix the two defects in 29_grpo_train.sh that the effective-batch work exposed.

DEFECT 1 -- the record could not be wrong out loud.
`"grad_accum": 4` was a literal, while the neighbouring fields already read their
value from the environment. Three completed records therefore state 4 for runs that
used 8, 8 and 32, and nothing in the record or the file could disclose it. It is
replaced by a read of GRAD_ACCUM plus a field saying where the value came from --
so a caller that forgets to export it produces a record that says so instead of a
record that quietly claims 4.

DEFECT 2 -- an unset tag silently shares another run's files.
`RUN_TAG="${RUN_TAG:-grpo}"` points the result and both logs at a fixed name, so a
caller that forgets to set it overwrites another run's artefacts rather than failing.
All four current callers set it explicitly, so requiring it changes no working path.

This must not be applied while any run is executing: bash reads a script
incrementally by byte offset, and editing a file mid-execution has already caused two
failures in this project. scripts/44_final_consolidation.sh applies it only after the
card is idle.
"""
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
P = R / "projects/wafer-defect-vlm/scripts/29_grpo_train.sh"
s = P.read_text(encoding="utf-8")

if "grad_accum_source" in s:
    print("already applied; nothing to do")
    sys.exit(0)

FIXES = []

# ---------------------------------------------------------------- defect 1
OLD1 = '''        "grad_accum": 4,'''
NEW1 = '''        "grad_accum": int(os.environ.get("GRAD_ACCUM", 4)),
        "grad_accum_source": (
            "GRAD_ACCUM environment variable"
            if "GRAD_ACCUM" in os.environ else
            "DEFAULT 4 -- GRAD_ACCUM was not exported by the caller, so this "
            "value is not evidence about how the run was actually configured"),'''
if s.count(OLD1) != 1:
    sys.exit(f"grad_accum anchor appears {s.count(OLD1)} times, expected 1")
s = s.replace(OLD1, NEW1)
FIXES.append("grad_accum read from GRAD_ACCUM, with a field naming its source")

# ---------------------------------------------------------------- defect 2
OLD2 = 'RUN_TAG="${RUN_TAG:-grpo}"'
NEW2 = ('RUN_TAG="${RUN_TAG:?RUN_TAG must be set: it names the result file and both '
        'logs, and a fixed default lets a later run overwrite the artefacts of an '
        'earlier one instead of failing}"')  # no apostrophe: bash parses the
# `:?` word with quoting rules, and one inside it opens a string the file
# never closes -- which is how the third seed's run was lost on 2026-09-16.
if s.count(OLD2) != 1:
    sys.exit(f"RUN_TAG anchor appears {s.count(OLD2)} times, expected 1")
s = s.replace(OLD2, NEW2)
FIXES.append("RUN_TAG is required rather than defaulting to a shared name")

OLD3 = 'RUN="${RUN_TAG:-grpo}" IS_LEVEL="$IS_LEVEL" LEARNING_RATE="$LEARNING_RATE" \\'
NEW3 = 'RUN="$RUN_TAG" IS_LEVEL="$IS_LEVEL" LEARNING_RATE="$LEARNING_RATE" \\'
if s.count(OLD3) != 1:
    sys.exit(f"RUN= anchor appears {s.count(OLD3)} times, expected 1")
s = s.replace(OLD3, NEW3)
FIXES.append("the inner RUN= no longer carries a second copy of the default")

# The file is bash with embedded python; check the python blocks still parse by
# compiling each heredoc, and check bash syntax on the whole file.
# Named after the file rather than a shared /tmp path: a redirected run of this
# patch would otherwise restore from (or overwrite) another copy's backup.
BACKUP = str(P) + ".bak-patch75"
shutil.copy(P, BACKUP)
if "GRAD_ACCUM" not in s:
    sys.exit("GRAD_ACCUM disappeared")
P.write_text(s, encoding="utf-8")

r = subprocess.run(["bash", "-n", str(P)], capture_output=True, text=True)
print(f"bash -n: {r.returncode} {r.stderr.strip()[:200]}")
if r.returncode != 0:
    # This printed only, and on 2026-09-16 it printed a failure here, its
    # caller logged `ok`, and the wrapper it had just written could not be
    # read: the next launch died on line 136 and the card went idle. The
    # embedded-python check below has always exited on failure; this one is
    # now the same kind of check, because a check that cannot stop anything
    # is a log line.
    shutil.copy(BACKUP, P)
    sys.exit(f"the patched 29_grpo_train.sh fails `bash -n`; restored:\n{r.stderr}")

# Every embedded python heredoc must still be valid.
blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", s, re.S)
ok = 0
for i, b in enumerate(blocks):
    try:
        ast.parse(b)
        ok += 1
    except SyntaxError as e:
        sys.exit(f"embedded python block {i} no longer parses: {e}")
print(f"embedded python blocks parsed: {ok}/{len(blocks)}")
print()
for f in FIXES:
    print(f"  - {f}")
