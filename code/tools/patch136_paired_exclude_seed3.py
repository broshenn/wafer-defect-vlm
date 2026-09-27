"""The third draw has to be a decision in every record that enumerates runs, not just
in the one the seed-variance tool reads.

`tools/paired_significance.py` guards itself against a run landing and changing nothing
in its record: every `*__report.json` on disk must be one of its runs, one of its
optional runs, or a named exclusion, and anything else makes it exit without writing.
The second seed was given an entry with its reason when it was launched; the third
needs the same, and the moment it needs it is the moment its eval writes the report --
which is inside queue 43's landing, whose step 4 is exactly this tool. A missing entry
there does not produce a slightly narrow record: it makes paired_significance exit
non-zero, and a landing that stops at step 4 leaves the day's chain without its
comparison tables.

The reason recorded here is the one that distinguishes the third draw from the second:
the second seed measures whether a difference replicates, the third measures whether
the difference is stable -- whether it is a spread. Neither is a new cell, so neither
changes a single contrast in this record, and both are rows in the comparison tables
rather than columns in LIMITATIONS.md's.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/paired_significance.py"

OLD = '''EXCLUDE = {
    "qwen35_9b_grpo_lr1e5_seed3408":
        "the second seed of GRPO_G4_lr1e5. Same configuration, so it is not a new "
        "cell in any contrast here; its purpose is the replication check in "
        "tools/seed_variance.py, and it carries a column of its own in the "
        "comparison tables.",
}'''

NEW = '''EXCLUDE = {
    "qwen35_9b_grpo_lr1e5_seed3408":
        "the second seed of GRPO_G4_lr1e5. Same configuration, so it is not a new "
        "cell in any contrast here; its purpose is the replication check in "
        "tools/seed_variance.py, and it carries a column of its own in the "
        "comparison tables.",
    "qwen35_9b_grpo_lr1e5_seed3409":
        "the third draw of the same cell. Not a new cell either: it changes no "
        "contrast in this record, and no contrast here can answer the question it "
        "exists for. One difference is not a spread, so tools/seed_variance.py reports "
        "this run as the third point of the range rather than as a second learning "
        "rate. Like the second seed it is a row in the comparison tables and not a "
        "column in LIMITATIONS.md's, which the landing runs with --no-column so that "
        "one configuration is not counted twice.",
}'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/paired_significance.py.bak-patch136")
if s.count(OLD) != 1:
    sys.exit(f"the EXCLUDE anchor matches {s.count(OLD)} times; nothing written")
TOOL.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/paired_significance.py.bak-patch136", TOOL)
    sys.exit(f"the patched paired_significance.py does not compile, restored:\n{r.stderr}")
print("paired_significance.py: the third draw is declared an exclusion, with its reason")

# Run it, and read the two lines that say whether the record still accounts for
# everything on disk. It writes its record; that is what the landing's step 4 does with
# it too, and it is regenerated from the reports each time, so a run here is safe.
r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=str(ROOT))
print(f"\n--- paired_significance.py: exit {r.returncode} ---")
for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-12:]:
    print("  " + ln[:170])
