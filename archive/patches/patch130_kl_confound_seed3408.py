"""Declare run 42 in `tools/kl_length_confound.py`'s RUNS, then regenerate the record.

The tool already includes a finished run it does not name -- and says so, in a NOTE that
asks for exactly this edit: *"Add it to RUNS so the record's order is a decision, not a
sort."* Until that edit is made, run 42 enters the record by directory order, which is a
fact about `sorted(glob)` rather than a decision by anyone. It is the same distinction
section 9 makes about the run set being enumerated rather than hand-listed: enumeration
finds the run, and a declaration says the run was meant to be there.

Why this has to happen before run 42's landing and not after: `sync_counts` reads this
record for the two sentences that quote it (the idle-step total in 8.9 and the per-value
check count in 9), and the landing's checker reads the document. Regenerating afterwards
would leave the landing green over a record with eight runs while nine exist -- the
counts consistent with a record that is stale, which is this document's recurring defect
reached from the inside.

It is appended rather than inserted beside `grpo`, its first seed, because this record's
run order is the order `sync_counts` writes the per-run idle list out in. Inserting it
mid-list would rewrite eight numbers to say the same thing in a different order, and the
only visible consequence would be a diff nobody can check. It does not match the
cell-pairing pattern (`_gNN_lrXX`), so the loop that builds learning-rate pairs skips it
by design: two seeds are one cell, not two learning rates.
"""
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/kl_length_confound.py"
REC = ROOT / "outputs/reports/kl_length_confound.json"

OLD = '''        "qwen35_9b_gspo_g4_lr1e5", "qwen35_9b_gspo_g32_lr1e5"]'''
NEW = '''        "qwen35_9b_gspo_g4_lr1e5", "qwen35_9b_gspo_g32_lr1e5",
        # The second seed of GRPO G=4 lr 1e-5 (queue 42). Appended, not grouped with
        # `grpo`, its first seed: this list's order is the order `sync_counts` writes
        # the per-run idle list out in, so inserting it mid-list would rewrite eight
        # numbers to say the same thing in a different order. It carries no `_gNN_lrXX`
        # shape, so the cell-pairing loop below skips it -- two seeds are one cell.
        "qwen35_9b_grpo_lr1e5_seed3408"]'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/kl_length_confound.py.bak-patch130")
if s.count(OLD) != 1:
    sys.exit(f"the RUNS anchor matches {s.count(OLD)} times; nothing written")
TOOL.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/kl_length_confound.py.bak-patch130", TOOL)
    sys.exit(f"kl_length_confound.py does not compile; restored:\n{r.stderr}")
print("kl_length_confound.py: run 42 declared in RUNS.")

before = json.loads(REC.read_text(encoding="utf-8"))
shutil.copy2(REC, "/tmp/kl_length_confound.json.bak-patch130")
print(f"record before: {len(before['runs'])} run(s), "
      f"total idle steps {sum(v['kl_identity']['idle_steps_tested'] for v in before['runs'].values())}")

r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=str(ROOT))
print(f"\n--- kl_length_confound.py: exit {r.returncode} ---")
for ln in (r.stdout or "").splitlines():
    print("  " + ln[:180])
if r.stderr.strip():
    print("  stderr:")
    for ln in r.stderr.strip().splitlines():
        print("    " + ln[:180])
if r.returncode != 0:
    shutil.copy2("/tmp/kl_length_confound.json.bak-patch130", REC)
    print("record restored from the backup; the declaration stands.")

after = json.loads(REC.read_text(encoding="utf-8"))
per = [v["kl_identity"]["idle_steps_tested"] for v in after["runs"].values()]
print(f"\nrecord after: {len(after['runs'])} run(s) {list(after['runs'])}")
print(f"per run: {per}  total {sum(per)}")
bad = [t for t, v in after["runs"].items() if not v["kl_identity"]["holds"]]
print(f"runs whose kl identity does NOT hold: {bad or 'none'}")
