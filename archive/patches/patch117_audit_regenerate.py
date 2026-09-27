"""Two last facts, and each had a different reason for being unreadable.

**1. Section 9's audit list is a claim about another tool's output.** The sentence says
the tool verifies every rate and lists them 「按该工具 `RUNS` 的顺序」. It is not a
sentence about the record -- it is a sentence about what `tools/idle_step_table.py`
prints, so the honest way to write it is to run that tool and quote it. This does, which
is why the count and the list here cannot drift from the tool independently: the tool's
own 「N run(s) verified」 line supplies the numeral and its own cell row supplies the
list. The checker reads the same two things from the tool's source and the reports, so a
landing that the tool picks up shows up on both sides.

**2. One comparison was over-strict, and the document was right.** 5.2.2 names which
pairs lr 5e-5 raises the idle-step share in, and the item compares the document's list
against `sorted(_raised)` -- sorted on one side only. With a single pair that is the same
thing; with two it demands that the document's prose be in lexicographic order, which the
sentence never claimed. The names in it are read off the four-pair enumeration directly
above it, and reading them in that order is the readability the enumeration exists for.
So the lambda sorts both sides: the item is about *which* pairs, and now says so.

The tool also reports one thing that is not this patch's business, and it is worth
recording because it is the RUN_TAG defect seen from the outside: a log named
`29_qwen35_9b_grpo_lr1e5_seed3408_train.log` sits on disk undeclared, because run 42 is
still writing it under a name its `RUNS` entry does not list. The tool is designed to
report exactly that rather than skip it, and it did. It is fixed after pid 3699 exits,
with the `29_grpo_train.sh` edits, since a run still reading that script cannot have it
edited underneath it.
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
TOOL = ROOT / "tools/idle_step_table.py"
DB = "/tmp/LIMITATIONS.md.bak-patch117"
CB = "/tmp/check_quantified_claims.py.bak-patch117"

# ---------------------------------------------- 1. section 9, from the tool itself
r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True,
                   cwd=str(ROOT))
if r.returncode != 0:
    sys.exit(f"idle_step_table.py exited {r.returncode}; nothing written:\n{r.stderr[-800:]}")
tool_out = r.stdout

m_n = re.search(r"^(\d+) run\(s\) verified", tool_out, re.M)
m_row = re.search(r"^\| ([\d.]+% \| [\d.]+%(?: \| [\d.]+%)*) \|$", tool_out, re.M)
if not m_n or not m_row:
    sys.exit("the tool's output no longer carries the two lines this reads (「N run(s) "
             f"verified」 and the cell row); nothing written.\n---\n{tool_out[-900:]}")
n_verified = int(m_n.group(1))
rates = [x.strip() for x in m_row.group(1).split("|")]
if len(rates) != n_verified:
    sys.exit(f"the tool verifies {n_verified} run(s) but prints {len(rates)} cell(s); the "
             f"two lines mean different populations and this will not guess which is the "
             f"sentence's")

new_clause = ("（当前 " + str(n_verified) + " 个 run 全部一致，按该工具 `RUNS` 的顺序：\n  "
              + "、".join(rates) + "）")
old_clause = re.search(r"（当前 " + r"(?:\d+|[一二三四五六七八九十]+)" + r" 个 run 全部一致，"
                       r"按该工具 `RUNS` 的顺序：\s*\n?\s*[\d.]+%(?:\s*、\s*[\d.]+%)*）",
                       DOC.read_text(encoding="utf-8"))
if not old_clause:
    sys.exit("section 9's audit clause anchors 0 times; nothing written")
doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, DB)
if doc.count(old_clause.group(0)) != 1:
    sys.exit(f"section 9's audit clause anchors {doc.count(old_clause.group(0))} times; "
             f"nothing written")
doc = doc.replace(old_clause.group(0), new_clause, 1)
DOC.write_text(doc, encoding="utf-8")
print(f"section 9's audit list is quoted from the tool: {n_verified} run(s), "
      f"{'、'.join(rates)}")

# ------------------------------------------- 2. the over-strict comparison, sorted twice
C_OLD = '''        _norm = lambda s: [x.strip() for x in s.split("、") if x.strip()]
'''
C_NEW = '''        # Sorted on both sides. The item is about *which* pairs, and the `why` says so
        # ("a rewrite of the names"); the document reads them off the four-pair
        # enumeration directly above, which is the order the enumeration exists to give.
        # Sorting one side only demanded that the prose be in lexicographic order --
        # invisible while the split had one pair, and a demand the sentence never made.
        _norm = lambda s: sorted(x.strip() for x in s.split("、") if x.strip())
'''
c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
if c.count(C_OLD) != 1:
    shutil.copy2(DB, DOC)
    sys.exit(f"the _norm line appears {c.count(C_OLD)} times; nothing further written")
c = c.replace(C_OLD, C_NEW, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    shutil.copy2(DB, DOC)
    sys.exit(f"the patched checker does not compile; both files restored:\n{r.stderr}")
print("the split item now compares the names as the set of names it claims to be.")

print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
n_stale = 0
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        n_stale += 1
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
for ln in out[-6:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s) printed")
