"""One current-state sentence left unread: section 9's description of the idle-step audit.

It reads:

    （当前 7 个 run 全部一致，按该工具 `RUNS` 的顺序：
      36.00%、32.67%、38.00%、15.33%、42.00%、26.67%、33.33%）

"当前" makes it a claim about now, and both the count and the list move the moment a run
that `tools/idle_step_table.py` covers gets a training record -- which is what run 41
does, and what run 42 does again. Nothing read it: the audit's own sentence was the one
place where a number could go stale with the checker green.

The population is not the RL run set: the audit also covers the second seed, which has no
column in any table. So it is read from the tool rather than restated -- the tool's
`RUNS` list is parsed out of its source and its order is used, with the entries whose
record is not on disk dropped. That is the same list the sentence says it is quoting
("按该工具 `RUNS` 的顺序"), so if the tool and the sentence ever disagree about what the
audit covers, this finds it instead of a hand-kept copy drifting.

The count is a `count` (a numeral); the rate list is a `fact` (a run is appended to it,
not a numeral changed), so `--fix` cannot quietly bump the number while the list beside
it still names the old population.
"""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"

ANCHOR = '''
# ------------------------------------------------------------------------ verdict'''

BLOCK = '''
# --------------------- 9: the idle-step audit's own coverage, read off the tool
# The sentence in section 9 says how many runs the audit covers and lists their rates in
# that tool's RUNS order. The population is the tool's, not the tables': it includes the
# second seed, which has no column anywhere. So the list is parsed out of the tool's
# source and filtered to the entries whose record is on disk -- never restated here,
# because a second copy of it is a second thing that can drift.
_IDLE_TOOL = ROOT / "tools/idle_step_table.py"
P_IDLE_AUDIT = (r"（当前 (" + NUM + r") 个 run 全部一致，按该工具 `RUNS` 的顺序：\\s*"
                r"([\\d.]+%(?:\\s*、\\s*[\\d.]+%)*)）")
g = anchor("9: the idle-step audit's coverage", P_IDLE_AUDIT)
if g and _IDLE_TOOL.is_file():
    _entries = []
    for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
        if (isinstance(_node, ast.Assign) and _node.targets
                and getattr(_node.targets[0], "id", None) == "RUNS"):
            for _e in _node.value.elts:
                _lab, _stem = ast.literal_eval(_e)[0], ast.literal_eval(_e)[1]
                _entries.append((_lab, _stem if _stem.endswith(".json")
                                 else _stem + ".json"))
    _covered = []
    for _lab, _stem in _entries:
        _p = REP / _stem
        if not _p.is_file():
            continue
        _v = json.loads(_p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
        if isinstance(_v, (int, float)):
            _covered.append((_lab, _v))
    if _covered:
        cmp("9: the idle-step audit's coverage (count)", cn2int(g.group(1)),
            len(_covered), kind="count", pattern=P_IDLE_AUDIT, group=1)
        cmp("9: the rates the audit lists, in the tool's RUNS order",
            [float(x) for x in re.split(r"\\s*、\\s*", g.group(2).rstrip("%").replace(
                "%", "")) if x],
            [round(100 * _v, 2) for _, _v in _covered],
            why="a run the audit picked up is an entry appended to this list; the "
                "numeral alone would leave the list naming the old population")
        print(f"         the audit covers {len(_covered)} run(s): "
              + "、".join(f"{_l} {100 * _v:.2f}%" for _l, _v in _covered))

# ------------------------------------------------------------------------ verdict'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch105")

for what, cond in (("the insertion point", c.count(ANCHOR) == 1),
                   ("the audit item", "P_IDLE_AUDIT" not in c),
                   ("ast is not already imported", c.count("\nimport ast\n") == 0)):
    if not cond:
        sys.exit(f"{what} is not as expected; nothing written")

# `import ast` at the top, next to the other stdlib imports. Placed by anchor rather
# than by line number so it survives the file being edited elsewhere.
if "\nimport ast\n" not in c:
    for _imp in ("import json\n", "import re\n", "import sys\n", "import os\n"):
        if "\n" + _imp in c:
            c = c.replace("\n" + _imp, "\nimport ast\n" + _imp, 1)
            break
    else:
        sys.exit("could not find an import line to put `import ast` beside; "
                 "nothing written")
if c.count("\nimport ast\n") != 1:
    sys.exit("`import ast` is now present more than once; nothing written")

c = c.replace(ANCHOR, BLOCK, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch105", CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the idle-step audit's coverage is now read: population from the tool's own RUNS "
      "list, order included.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
for ln in r.stdout.strip().splitlines()[-13:]:
    print("  " + ln)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch105", CHECKER)
    sys.exit(f"\nthe checker is not green (exit {r.returncode}); RESTORED -- the item "
             f"disagrees with the document and the item is the thing to suspect")
print("\nchecker green: every current-state sentence that moves with the run set is now "
      "read from the records")
