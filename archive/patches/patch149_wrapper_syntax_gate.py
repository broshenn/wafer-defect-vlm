"""The wrapper would not parse, and the check that saw it printed instead of stopping.

At 17:36:14 tools/patch75_grpo_record.py rewrote `29_grpo_train.sh` and its own syntax
check reported the result:

    bash -n: 2 .../29_grpo_train.sh: line 136: unexpected EOF while looking for matching `'`

A minute later the driver logged `ok` for that step, queue 43 -- which had been waiting for
exactly those patches to finish -- launched the wrapper, and the third seed's training died
on the first line with the same error. The card has been idle since.

The bar is not `bash -n`'s message. It is `RUN_TAG="${RUN_TAG:?...another run's
artefacts..."}`, written by patch75 with an apostrophe inside the `:?` word: bash parses
that word with quoting rules, the apostrophe opens a string that the file never closes, and
everything after line 18 stops being shell at all. patch75 escaped it correctly as Python
(`'run\\'s artefacts'`) and produced a bash file that cannot be read.

Why nothing stopped it:

  * patch75 checks the embedded python blocks with `sys.exit` on failure, and checks bash
    syntax with `print`. One file, two checks, one enforcing -- and the advisory one was
    the one that mattered. The caller logged `ok`, which is what patch75's exit code said.
  * tools/patch118_grpo_train_fixes.py edits the same two lines and *does* restore on a
    failed `bash -n`. It stood down, correctly, because patch75 had already written the
    `grad_accum_source` field it looks for -- so the one check that would have caught this
    was unreachable by construction. Section 8 item 15's shape, one hour old and load
    bearing.

Three edits here: the wrapper's line is fixed, patch75's check stops instead of printing
(and restores from a backup named after the file, not a shared /tmp path), and patch118
checks the syntax in both branches so standing down can no longer turn the check off.

The enforcing path is exercised, not asserted: patch75 is run against a copy of the
pre-patch wrapper (the backup its own first run left in /tmp), once clean and once with a
syntax error injected, and the second run must restore the file byte for byte and exit
non-zero.
"""
import hashlib
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
PY = str(ROOT / "venvs/wafer/bin/python")
SH = ROOT / "projects/wafer-defect-vlm/scripts/29_grpo_train.sh"
P75 = ROOT / "tools/patch75_grpo_record.py"
P118 = ROOT / "tools/patch118_grpo_train_fixes.py"

BROKEN = ('RUN_TAG="${RUN_TAG:?RUN_TAG must be set: it names the result file and both '
          "logs, and defaulting it to a fixed string lets a new run overwrite another "
          "run's artefacts instead of failing}\"")
FIXED = ('RUN_TAG="${RUN_TAG:?RUN_TAG must be set: it names the result file and both '
         "logs, and a fixed default lets a later run overwrite the artefacts of an "
         "earlier one instead of failing}\"")

# ------------------------------------------------------------------ 1. the wrapper
s = SH.read_text(encoding="utf-8")
if s.count(BROKEN) != 1:
    sys.exit(f"the broken RUN_TAG line matches {s.count(BROKEN)} times; nothing written")
shutil.copy2(SH, "/tmp/29_grpo_train.sh.bak-patch149")
SH.write_text(s.replace(BROKEN, FIXED, 1), encoding="utf-8")
r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/29_grpo_train.sh.bak-patch149", SH)
    sys.exit(f"the wrapper still does not parse, restored:\n{r.stderr}")
p = subprocess.run([PY, "-c",
                    "import pathlib, sys;"
                    "t = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8');"
                    "print([l for l in t.splitlines()"
                    " if l.startswith('RUN_TAG=')][0][:120])",
                    str(SH)], capture_output=True, text=True)
print("29_grpo_train.sh parses again; its RUN_TAG line is now:")
print("  " + p.stdout.strip()[:130])

# ------------------------------------------------- 2. patch75's text and its gate
t = P75.read_text(encoding="utf-8")
OLD_NEW2 = ("NEW2 = ('RUN_TAG=\"${RUN_TAG:?RUN_TAG must be set: it names the result file "
            "and both '\n"
            "        'logs, and defaulting it to a fixed string lets a new run overwrite "
            "another '\n"
            "        'run\\'s artefacts instead of failing}\"')\n")
NEW_NEW2 = ("NEW2 = ('RUN_TAG=\"${RUN_TAG:?RUN_TAG must be set: it names the result file "
            "and both '\n"
            "        'logs, and a fixed default lets a later run overwrite the artefacts "
            "of an '\n"
            "        'earlier one instead of failing}\"')  # no apostrophe: bash parses the\n"
            "# `:?` word with quoting rules, and one inside it opens a string the file\n"
            "# never closes -- which is how the third seed's run was lost on 2026-09-16.\n")
OLD_BAK = 'shutil.copy(P, "/tmp/29_before_fix.sh")\n'
NEW_BAK = ('# Named after the file rather than a shared /tmp path: a redirected run of this\n'
           '# patch would otherwise restore from (or overwrite) another copy\'s backup.\n'
           'BACKUP = str(P) + ".bak-patch75"\n'
           'shutil.copy(P, BACKUP)\n')
OLD_CHK = ('r = subprocess.run(["bash", "-n", str(P)], capture_output=True, text=True)\n'
           'print(f"bash -n: {r.returncode} {r.stderr.strip()[:200]}")\n')
NEW_CHK = ('r = subprocess.run(["bash", "-n", str(P)], capture_output=True, text=True)\n'
           'print(f"bash -n: {r.returncode} {r.stderr.strip()[:200]}")\n'
           'if r.returncode != 0:\n'
           '    # This printed only, and on 2026-09-16 it printed a failure here, its\n'
           '    # caller logged `ok`, and the wrapper it had just written could not be\n'
           '    # read: the next launch died on line 136 and the card went idle. The\n'
           '    # embedded-python check below has always exited on failure; this one is\n'
           '    # now the same kind of check, because a check that cannot stop anything\n'
           '    # is a log line.\n'
           '    shutil.copy(BACKUP, P)\n'
           '    sys.exit(f"the patched 29_grpo_train.sh fails `bash -n`; restored:\\n'
           '{r.stderr}")\n')
for what, old, new in (("patch75's replacement text", OLD_NEW2, NEW_NEW2),
                       ("patch75's backup path", OLD_BAK, NEW_BAK),
                       ("patch75's bash -n check", OLD_CHK, NEW_CHK)):
    if t.count(old) != 1:
        sys.exit(f"{what} matches {t.count(old)} times; nothing written")
    t = t.replace(old, new, 1)
shutil.copy2(P75, "/tmp/patch75_grpo_record.py.bak-patch149")
P75.write_text(t, encoding="utf-8")
r = subprocess.run([PY, "-m", "py_compile", str(P75)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/patch75_grpo_record.py.bak-patch149", P75)
    sys.exit(f"the patched patch75 does not compile, restored:\n{r.stderr}")
print("\npatch75_grpo_record.py: the bash check stops and restores; the apostrophe is gone")

# --------------------------------------- 3. patch118 checks syntax in both branches
c = P118.read_text(encoding="utf-8")
OLD_118 = ('    print("29_grpo_train.sh: RUN_TAG now has no default, and grad_accum reads '
           'GRAD_ACCUM.")\n')
NEW_118 = (OLD_118 +
           '\n'
           '# Checked here, unconditionally, rather than only inside the branch that\n'
           '# writes the file. The stand-down above fires whenever patch75 has run first --\n'
           '# which is the normal order -- so the `bash -n` in the else branch cannot see a\n'
           '# syntax error introduced by patch75. On 2026-09-16 exactly that happened: the\n'
           '# one check that would have caught it was unreachable by construction.\n'
           '_syn = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True,\n'
           '                      text=True)\n'
           'if _syn.returncode != 0:\n'
           '    sys.exit("29_grpo_train.sh does not parse; something else wrote it:\\n"\n'
           '             + _syn.stderr)\n')
if c.count(OLD_118) != 1:
    sys.exit(f"patch118's stand-down tail matches {c.count(OLD_118)} times; nothing written")
shutil.copy2(P118, "/tmp/patch118_grpo_train_fixes.py.bak-patch149")
P118.write_text(c.replace(OLD_118, NEW_118, 1), encoding="utf-8")
r = subprocess.run([PY, "-m", "py_compile", str(P118)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/patch118_grpo_train_fixes.py.bak-patch149", P118)
    sys.exit(f"the patched patch118 does not compile, restored:\n{r.stderr}")
print("patch118_grpo_train_fixes.py: the syntax check now runs in both branches")

# ------------------------------- 4. exercise patch75's gate, on copies, both directions
T = pathlib.Path("/tmp/p75test")
T.mkdir(parents=True, exist_ok=True)
PRE = pathlib.Path("/tmp/29_before_fix.sh")
if not PRE.is_file():
    sys.exit("the pre-patch wrapper backup is gone; the gate cannot be exercised")
src = P75.read_text(encoding="utf-8").replace(
    'P = R / "projects/wafer-defect-vlm/scripts/29_grpo_train.sh"',
    f'P = Path("{T}/29_grpo_train.sh")', 1)
(T / "p75.py").write_text(src, encoding="utf-8")


def run_case(label, mutate):
    target = T / "29_grpo_train.sh"
    target.write_text(PRE.read_text(encoding="utf-8") + mutate, encoding="utf-8")
    before = hashlib.md5(target.read_bytes()).hexdigest()
    r = subprocess.run([PY, str(T / "p75.py")], capture_output=True, text=True,
                       cwd=str(T))
    after = hashlib.md5(target.read_bytes()).hexdigest()
    out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    print(f"\n--- patch75 [{label}]: exit {r.returncode} ---")
    for ln in out[-3:]:
        print("  " + ln[:150])
    print(f"  file restored byte-for-byte: {before == after}")
    return r.returncode, before == after


rc_clean, restored_clean = run_case("the wrapper as it was before the fix", "")
if rc_clean != 0 or not restored_clean:
    sys.exit("patch75 no longer applies cleanly to the pre-patch wrapper")
body = (T / "29_grpo_train.sh").read_text(encoding="utf-8")
if "run's artefacts" in body or "run's artefacts" in body:
    sys.exit("patch75 still writes an apostrophe into the `:?` word")
print("  and the file it wrote has no apostrophe in the `:?` word")

rc_bad, restored_bad = run_case("the same wrapper with a syntax error injected",
                                "\nif [ 1 -eq 1 ]; then\n")
if rc_bad == 0:
    sys.exit("patch75 still exits 0 on a file that does not parse: the gate is not "
             "enforcing and the run would be lost the same way")
if not restored_bad:
    sys.exit("patch75 exited non-zero but left the broken file in place")
print("\nthe gate stops on a broken file and restores it; on a good one it proceeds")
