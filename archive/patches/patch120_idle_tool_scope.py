"""The block added by patch119 read a name that is defined forty lines further down.

    NameError: name '_IDLE_TOOL' is not defined

`_IDLE_TOOL` is assigned just before section 9's audit, which sits after the 5.2.4
blocks; Python has no forward reference for a module-level name that has not executed
yet. The insert therefore gets its own binding rather than moving the existing one:
moving it would change the line section 9 depends on for no reason, and the two agree
by construction because both are the same path spelled the same way.

The failure is worth one line because of how it presented: the checker exited 1 with
**zero** STALE items. Exit 1 means "a fact has moved" in this tool's vocabulary, so the
run looked like a document problem rather than a code problem, and the traceback was on
stderr while the STALE scan reads stdout. A tool whose exit code is part of its interface
cannot die partway and leave the code meaning what it always meant -- the crash and the
verdict share one integer. Section 8's own rule is that a tool is not trusted until it
has failed in a way someone checked; this is that failure, and the fix is the name.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
CB = "/tmp/check_quantified_claims.py.bak-patch120"

OLD = '''    _stem = {}
    if _IDLE_TOOL.is_file():'''
NEW = '''    _stem = {}
    # `_IDLE_TOOL` is assigned below, just before section 9's audit, so it does not
    # exist in this scope yet -- a module-level name is not forward-referenced. Binding
    # it here rather than moving the other one leaves section 9's line untouched.
    _IDLE_TOOL = ROOT / "tools/idle_step_table.py"
    if _IDLE_TOOL.is_file():'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
if c.count(OLD) != 1:
    sys.exit(f"the anchor matches {c.count(OLD)} times, expected 1; nothing written")
CHECKER.write_text(c.replace(OLD, NEW, 1), encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    sys.exit(f"does not compile; restored:\n{r.stderr}")

r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = (r.stdout or "").strip().splitlines()
print("--- the std superlative's items ---")
for ln in out:
    s = ln.strip()
    if ("std superlative" in s or "lowest std" in s or "holds it" in s
            or "RL groups in" in s or "reward_signal" in s):
        print("  " + s[:150])
n_stale = sum(1 for ln in out if ln.strip().startswith("STALE"))
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
print("--- tail ---")
for ln in out[-4:]:
    print("  " + ln.strip())
if r.stderr.strip():
    print("--- stderr ---")
    print(r.stderr.strip()[-700:])
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
