"""Both directions of patch75's syntax gate, and patch118's new unconditional check.

The gate is the thing that was missing when the third seed's run was lost, so it is
measured rather than described:

  * clean input  -> exit 0, the fix is written, and the `:?` word contains no apostrophe;
  * broken input -> exit non-zero and the file restored byte for byte.

The second case is the one that matters: patch75's check used to print and carry on, and
"carry on" is what left a wrapper that could not be read. The first case must *not* restore
-- restoring there would silently undo the fix and look like success, which is why the
assertion is inverted between the two.
"""
import hashlib
import pathlib
import subprocess
import sys

PY = "/root/autodl-fs/wafer-vlm/venvs/wafer/bin/python"
T = pathlib.Path("/tmp/p75test")
PRE = pathlib.Path("/tmp/29_before_fix.sh")
TARGET = T / "29_grpo_train.sh"
P75 = T / "p75.py"

for p in (PRE, P75):
    if not p.is_file():
        sys.exit(f"{p} is missing; the earlier patch run did not leave what this needs")

if "grad_accum_source" in PRE.read_text(encoding="utf-8"):
    sys.exit("the backup is already patched; it cannot serve as the pre-patch input")


def run_case(label, mutate, expect_restore):
    TARGET.write_text(PRE.read_text(encoding="utf-8") + mutate, encoding="utf-8")
    before = hashlib.md5(TARGET.read_bytes()).hexdigest()
    r = subprocess.run([PY, str(P75)], capture_output=True, text=True, cwd=str(T))
    after = hashlib.md5(TARGET.read_bytes()).hexdigest()
    out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    print(f"\n--- patch75 [{label}]: exit {r.returncode} ---")
    for ln in out[-3:]:
        print("  " + ln[:150])
    body = TARGET.read_text(encoding="utf-8")
    ok = True
    if expect_restore:
        if r.returncode == 0:
            print("  FAIL: exited 0 on a file that does not parse")
            ok = False
        if before != after:
            print("  FAIL: exited non-zero but did not restore the file")
            ok = False
        else:
            print("  restored byte-for-byte, exit non-zero")
    else:
        if r.returncode != 0:
            print("  FAIL: a valid wrapper was rejected")
            ok = False
        if before == after:
            print("  FAIL: nothing was written, so the fix was not applied")
            ok = False
        if "grad_accum_source" not in body:
            print("  FAIL: the written file has no grad_accum_source field")
            ok = False
        line = [l for l in body.splitlines() if l.startswith("RUN_TAG=")][0]
        if "'" in line:
            print(f"  FAIL: an apostrophe is still in the `:?` word: {line[:90]}")
            ok = False
        else:
            print("  the fix is written and the `:?` word has no apostrophe in it")
    return ok


a = run_case("the wrapper as it was before the fix", "", expect_restore=False)
b = run_case("the same wrapper with a syntax error injected",
             "\nif [ 1 -eq 1 ]; then\n", expect_restore=True)
print(f"\nclean case {'ok' if a else 'FAILED'}; broken case {'ok' if b else 'FAILED'}")
sys.exit(0 if (a and b) else 1)
