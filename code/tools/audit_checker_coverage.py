"""Which lines of the document does the certifying tool actually read?

The checker exits 0 when every item it has matches the records. That is not the same as
"the document has been verified": an item exists only for a sentence someone wrote one
for. A sentence carrying a count or a quantifier that no item anchors on is a sentence
that can go stale with the tool green -- section 8 item 9, and the failure this project
has now found three times (the idle-step 六组, the unread class list, this table).

So this measures the tool's coverage instead of assuming it. It runs the checker with
`re.search`/`re.fullmatch` wrapped, records the span of every successful match against
the document text, and reports the document's quantifier-bearing lines that no match
covers. Those are the places where a number can move unobserved.

How the spans are filtered: the wrapper only counts a hit when the searched string *is*
the document (same length, same opening). Identity will not do it -- the checker reads
the file for itself, so its `text` is a different object from this script's. The checker
also searches other strings (table lines, file paths, its own patterns), and a span from
one of those would otherwise be read as covering a line of the document it never touched.

What it does NOT do: judge whether a *covered* line's item is the right item. A pattern
can match a line and still not be about the number on it. This is an inventory of blind
spots, not a proof of coverage -- the distinction is the whole point of the exercise.

Exit status: 0 no unread quantifier line; 1 at least one, listed.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import re
import runpy
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
DOC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "LIMITATIONS.md")

text = DOC.read_text(encoding="utf-8")
LEAD = text[:200]


def _is_doc(s):
    """Is this the document text? By content -- see the note above about identity."""
    return isinstance(s, str) and len(s) == len(text) and s[:200] == LEAD


hits = []
_real_search, _real_fullmatch = re.search, re.fullmatch


def _search(pattern, string, flags=0):
    m = _real_search(pattern, string, flags)
    if m and _is_doc(string):
        hits.append(m.span())
    return m


def _fullmatch(pattern, string, flags=0):
    m = _real_fullmatch(pattern, string, flags)
    if m and _is_doc(string):
        hits.append(m.span())
    return m


re.search, re.fullmatch = _search, _fullmatch
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(CHECKER), run_name="__main__")
    code = 0
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else 0
finally:
    re.search, re.fullmatch = _real_search, _real_fullmatch

lines = text.split("\n")
covered = set()
for a, b in hits:
    covered.update(range(text.count("\n", 0, a), text.count("\n", 0, b) + 1))

print(f"the checker exited {code}")
print(f"document: {len(lines)} lines; successful matches against it: {len(hits)}")
print(f"lines covered by at least one match: {len(covered)}")
print()

# A quantifier is a number with a unit: 六个 run, 七列, 四句, 三条. Written either way --
# the document uses both (六个 in 5.2.5, "6 个 run" in section 9).
QUANT = re.compile(r"[0-9一二三四五六七八九十两]+\s*(个|组|列|句|条|张|行|次|种)")

# Lines inside the two eight-column tables are read -- structurally, by patch107's
# cell-by-cell comparison, not by a pattern. Counting them as uncovered would report the
# best-covered part of the document as a blind spot.
def table_lines():
    out = set()
    for i, l in enumerate(lines):
        if not l.startswith("| 指标 |") or i + 1 >= len(lines):
            continue
        if lines[i + 1].strip().strip("|").replace("-", "").replace("|", "").strip():
            continue
        if len(l.strip().strip("|").split("|")) - 1 != 8:
            continue
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            out.add(j)
            j += 1
    return out


TABLES = table_lines()

# History is not a claim about now. The document records its own earlier states on
# purpose (that is what makes it a limitations section rather than a brochure), and a
# sentence about what the first draft said cannot go stale. What matters is the rest:
# a quantifier over the *current* run set.
HIST = ("初稿", "原本", "原先", "原稿", "当时", "第一次", "首轮", "曾", "上一版", "已改",
        "已删", "在这之前", "此前")

# The nouns that make a quantifier move when a run lands. Drawn from the document's own
# vocabulary rather than invented, so the bucket is the sentences this project already
# treats as run-set-dependent.
RUNNOUN = ("run", "列", "组大小", "空转步", "加粗", "极值", "最高", "最低", "最差", "最好")


def classify(l):
    live = any(k in l for k in RUNNOUN)
    hist = any(k in l for k in HIST)
    return live, hist


unread = []
for i, l in enumerate(lines):
    if i in covered or i in TABLES:
        continue
    if not QUANT.search(l):
        continue
    unread.append((i + 1, l.strip()))

live = [(n, l) for n, l in unread if classify(l)[0] and not classify(l)[1]]
hist = [(n, l) for n, l in unread if classify(l)[1]]
other = [(n, l) for n, l in unread if not classify(l)[0] and not classify(l)[1]]

print(f"--- quantifier lines no item reads: {len(unread)} of {len(lines)} lines ---")
print(f"    {len(live):>3} carry a run-set-dependent noun and no past-tense marker "
      f"-- a numeral that can move with the tool green")
print(f"    {len(hist):>3} are prose about the document's own earlier states "
      f"(cannot go stale by construction)")
print(f"    {len(other):>3} are neither")
print()
print("--- the live ones, listed ---")
for n, l in live:
    print(f"  {n:>4}| {l[:150]}")
if not live:
    print("  (none)")
print()
print("An entry here is a candidate, not a defect: the sentence may be about the reward")
print("table, or about a quantity that does not move when a run lands. But it is where a")
print("stale numeral would sit, and the tool would still exit 0.")
sys.exit(1 if live else 0)
