"""44 was already waiting when queue 43 was launched, and its waiter did not know it.

`44_final_consolidation.sh` waits for an idle card before it rewrites LIMITATIONS.md,
comparison.json and FINAL_REPORT.md. The wait is `tools/wait_for_idle.py`, and its rule
for queue drivers is a two-entry hand-written list:

    EXACT = ["bash projects/.../41_queue_gspo_g32_lr1e5.sh",
             "bash projects/.../42_queue_grpo_seed2.sh"]

Queue 43 (`43_queue_grpo_seed3.sh`, the third seed) was launched at 17:18 -- while 44
was running, waiting. Its training matches the SUBSTR rules, so 44 will wait through
the training and the evaluation. But the queue's *landing* -- step 6, which calls
`land_run.sh` and therefore `sync_counts.py` and the quantified-claims checker -- is
plain python and shell with none of those signatures. So the moment the evaluation
ends, 44's waiter would see an idle card and start rewriting the same document the
landing is rewriting: two writers, whichever finishes second wins, and the winner is
not the one whose sentences were checked against the records. That is exactly the race
`land42_driver.sh` exists to prevent for run 42 -- its comment says "Two writers to one
document is a race with no useful outcome" -- and it is back, because the fix was a
list.

The list is the defect. It is correct when written and silently incomplete afterwards,
which is section 8 item 15's shape applied to a waiter: a mechanism that exists and
does not run, with the only symptom being a race that resolves itself silently. So the
queue drivers are enumerated from disk instead -- every `*_queue_*.sh` under the
project's scripts directory, in both the forms a launcher may use (absolute, and
relative to the wafer root) -- and a new queue script is waited for without anyone
remembering to add it.

Equality is kept as the matcher, deliberately: a launcher's command line can contain a
script's path as an *argument* (this project pushes scripts as base64 blobs and copies
them by path), and only a process that is the driver has that path as its whole command
line. That is the same distinction the module's docstring draws for SUBSTR.

Test: the patched module is imported and its `busy()` is called, so the derivation is
shown to produce the running queue's command line rather than assumed to.
"""
import importlib.util
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/wait_for_idle.py"

OLD = '''EXACT = [
    "bash projects/wafer-defect-vlm/scripts/41_queue_gspo_g32_lr1e5.sh",
    "bash projects/wafer-defect-vlm/scripts/42_queue_grpo_seed2.sh",
]'''

NEW = '''# Enumerated from disk rather than hand-written. The hand-written version held two
# command lines and was correct when written; queue 43 was launched while 44 was already
# waiting, so 44 would have started rewriting LIMITATIONS.md during that queue's own
# landing -- the race this waiter exists to prevent, reintroduced by a list that went
# stale. Both forms are generated because the launcher may use either, and a queue
# script added tomorrow is covered without anyone remembering this line.
QUEUE_DIR = pathlib.Path(__file__).resolve().parent.parent / "projects/wafer-defect-vlm/scripts"
EXACT = [
    "bash projects/wafer-defect-vlm/scripts/41_queue_gspo_g32_lr1e5.sh",
    "bash projects/wafer-defect-vlm/scripts/42_queue_grpo_seed2.sh",
]
for _q in sorted(QUEUE_DIR.glob("*_queue_*.sh")):
    EXACT.append(f"bash {_q}")
    try:
        EXACT.append(f"bash {_q.relative_to(QUEUE_DIR.parent.parent.parent)}")
    except ValueError:
        pass
EXACT = list(dict.fromkeys(EXACT))'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/wait_for_idle.py.bak-patch139")
if s.count("import argparse") != 1:
    sys.exit("wait_for_idle.py is not shaped as expected (import argparse); nothing "
             "written")
if "import pathlib" not in s and "\nimport pathlib" not in s:
    s = s.replace("import json", "import json\nimport pathlib", 1)
if s.count(OLD) != 1:
    sys.exit(f"the EXACT anchor matches {s.count(OLD)} times; nothing written")
TOOL.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/wait_for_idle.py.bak-patch139", TOOL)
    sys.exit(f"the patched wait_for_idle.py does not compile, restored:\n{r.stderr}")
print("wait_for_idle.py: queue drivers are enumerated from disk")

# ------------------------------------------------------------------ and run it
spec = importlib.util.spec_from_file_location("wfi", TOOL)
wfi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wfi)
print(f"\nEXACT now holds {len(wfi.EXACT)} command line(s):")
for c in wfi.EXACT:
    print(f"  {c}")
hits = wfi.busy()
print(f"\nbusy() right now: {len(hits)} match(es)")
for pid, why, c in hits[:6]:
    print(f"  {pid}  {why}  {c[:100]}")
queues = [h for h in hits if h[1].startswith("exact:")]
print("\nqueue drivers recognised as busy: "
      + (", ".join(h[2].split("/")[-1] for h in queues) or "NONE"))
