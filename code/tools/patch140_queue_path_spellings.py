"""The enumeration resolved a symlink, so it derived a path no process has.

patch139 replaced the hand-written queue list in `wait_for_idle.py` with an enumeration
from disk. It used `Path(__file__).resolve()`, and `/root/autodl-fs` is a symlink to
`/autodl-fs/data` on this machine, so the paths it generated were
`bash /autodl-fs/data/wafer-vlm/projects/.../43_queue_grpo_seed3.sh` -- while the queue
process's own command line is `bash /root/autodl-fs/wafer-vlm/projects/...`. The test
printout said it plainly: `busy()` recognised queue 42 (launched with a relative path)
and did not recognise queue 43, which is the one that had just been launched. A waiter
that enumerates the right scripts under the wrong spelling is the same failure as the
list it replaced, with more code around it.

That is why the check is run inside the patch rather than described in it: `busy()` is
called on the live process table and the result is printed, so "the queue is now waited
for" is a measurement rather than a claim. The fix keeps both spellings -- the literal
one from `__file__`, which is what a launcher using /root produces, and the resolved
one -- because either can be the form a shell was invoked with.
"""
import importlib.util
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/wait_for_idle.py"

OLD = '''QUEUE_DIR = pathlib.Path(__file__).resolve().parent.parent / "projects/wafer-defect-vlm/scripts"'''

NEW = '''# Both spellings: `/root/autodl-fs` is a symlink to `/autodl-fs/data` here, and a shell
# invoked through either one carries that spelling in its command line. patch139 used
# `resolve()` alone, derived `/autodl-fs/data/...` while queue 43 was running as
# `/root/autodl-fs/...`, and recognised nothing -- a waiter that enumerates the right
# script under the wrong name is the list-shaped failure it was meant to replace.
QUEUE_DIRS = []
for _f in (pathlib.Path(__file__), pathlib.Path(__file__).resolve()):
    _d = _f.parent.parent / "projects/wafer-defect-vlm/scripts"
    if _d not in QUEUE_DIRS:
        QUEUE_DIRS.append(_d)'''

OLD2 = '''for _q in sorted(QUEUE_DIR.glob("*_queue_*.sh")):
    EXACT.append(f"bash {_q}")
    try:
        EXACT.append(f"bash {_q.relative_to(QUEUE_DIR.parent.parent.parent)}")
    except ValueError:
        pass'''

NEW2 = '''for _d in QUEUE_DIRS:
    for _q in sorted(_d.glob("*_queue_*.sh")):
        EXACT.append(f"bash {_q}")
        try:
            EXACT.append(f"bash {_q.relative_to(_d.parent.parent)}")
        except ValueError:
            pass'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/wait_for_idle.py.bak-patch140")
for what, old, new in (("the directory", OLD, NEW), ("the loop", OLD2, NEW2)):
    if s.count(old) != 1:
        sys.exit(f"{what}: the anchor matches {s.count(old)} times; nothing written")
    s = s.replace(old, new, 1)
TOOL.write_text(s, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/wait_for_idle.py.bak-patch140", TOOL)
    sys.exit(f"the patched wait_for_idle.py does not compile, restored:\n{r.stderr}")

spec = importlib.util.spec_from_file_location("wfi", TOOL)
wfi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wfi)
print("EXACT now holds:")
for c in wfi.EXACT:
    print(f"  {c}")
hits = wfi.busy()
print(f"\nbusy() right now: {len(hits)} match(es)")
for pid, why, c in hits[:6]:
    print(f"  {pid}  {why}  {c[:110]}")
queues = [h for h in hits if h[1].startswith("exact:")]
names = [h[2].split("/")[-1] for h in queues]
print("\nqueue drivers recognised as busy: " + (", ".join(names) or "NONE"))
if "43_queue_grpo_seed3.sh" not in names:
    sys.exit("queue 43 is running and is still not recognised; the evening's race is "
             "not fixed")
print("queue 43 is recognised: 44 will wait through its landing rather than racing it")
