"""`lr_cells` yields three values; this loop unpacked four, and run 41 made it run.

    File "tools/paired_significance.py", line 460, in <module>
      for _algo, g, a, b in lr_cells(correct):
  ValueError: not enough values to unpack (expected 4, got 3)

`lr_cells` returns `(label, lr1e-5 run, lr5e-5 run)` and says so in its own docstring.
Two callers in the file unpack three (lines 300 and 625). The block at 460 was written
ahead of run 41 -- it is guarded by `if "GSPO_G32_lr1e5" in correct:`, and one of its
messages says "G=32 at lr 1e-5 is run by queue 41 and is not in this record" -- so it
sat unexecuted from the day it was written until the moment queue 41 landed. Dead code
with a typo in it is not a typo; it is a defect that waits for the condition that makes
it live, and the condition here was the run the whole block was written for.

The group size is not lost: `lr_cells` puts it in the label ("GSPO G=32"), which is what
the label is for. It is read from there and converted to int, because the block formats
it with `f"G={g:<3d}"`.

The failure also has a shape worth recording, because it is the opposite of the one this
document keeps finding. The landing script had already added the column to the four
tables when step 4 died, so the document was left **half-landed**: the tables carried a
tenth column while the paired record still described eight runs. `land_run.sh`'s own
header warns about exactly this ("a lander that added the column and then died on a
missing patch script would leave the document half-updated, which is worse than not
starting"), and its ordering puts the cheapest, most-reversible step first and the record
regeneration second -- so the crash landed in the gap between them. What made it
recoverable rather than corrupting is that `add_run_columns.py` refuses to add a column
that is already present: the resume has to go through steps 4-6, not through step 3.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/paired_significance.py"
BAK = "/tmp/paired_significance.py.bak-lr_cells"

OLD = '''    for _algo, g, a, b in lr_cells(correct):
        if _algo != "GSPO":
            continue
'''

NEW = '''    # `lr_cells` yields (label, lr1e-5 run, lr5e-5 run) -- three values, which is what
    # the two other callers in this file unpack. This loop unpacked four, expecting the
    # group size as a fourth element. It ran for the first time when run 41 gave GSPO a
    # second group size, and raised ValueError, so the record did not write. The group
    # size is in the label, which is what the label is for.
    for _label, a, b in lr_cells(correct):
        _algo, _gs = _label.split(" G=")
        g = int(_gs)
        if _algo != "GSPO":
            continue
'''

t = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, BAK)
if t.count(OLD) != 1:
    sys.exit(f"the loop header appears {t.count(OLD)} times, expected 1; nothing written")

t = t.replace(OLD, NEW, 1)
TOOL.write_text(t, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, TOOL)
    sys.exit(f"the patched tool does not compile, restored:\n{r.stderr}")

print("paired_significance.py: the group size is read from the label lr_cells already "
      "returns.")
print("\n--- re-running the paired record (this is step 4 of the landing) ---")
r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=str(ROOT))
print(r.stdout[-2600:] if len(r.stdout) > 2600 else r.stdout)
if r.stderr.strip():
    print("--- stderr ---")
    print(r.stderr.strip()[-1200:])
print(f"\npaired_significance exit {r.returncode}")
if r.returncode != 0:
    print("STILL FAILING. The patch is in; the failure is a different one, above.")
