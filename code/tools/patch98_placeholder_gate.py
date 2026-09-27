"""Make the number audit fail on an unfilled template field, and record the defect.

Two sentences in LIMITATIONS.md were shipped with their format fields unsubstituted
(section 5.2.3 and 5.2.5, both quoting the single-variable IS pair). They are now
filled in from the record by patch96. What is left is to make the class impossible
to repeat silently.

Why this belongs in the audit and not in a one-off patch: the audit is the only tool
that reads every number in all three documents on every consolidation, and an
unfilled field is not a wrong number -- it is the absence of a number in a sentence
that reads as if it had one. Every other tool here compares digits against records;
a field has no digits to compare. So the audit gains a second, cruder check: does
the document contain a template field at all. There is no legitimate prose form
that looks like one, so a hit is not "review this", it is a failure -- and the audit
now exits non-zero on it, which makes tools/sync_counts.py (which parses the audit's
summary and refuses to run if the audit fails) refuse too.

Also records the defect as section 8 item 14.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
AUDIT = ROOT / "tools/audit_report_numbers.py"
DOC = ROOT / "LIMITATIONS.md"

# ------------------------------------------------------------------ 1. the audit
a = AUDIT.read_text(encoding="utf-8")
shutil.copy2(AUDIT, "/tmp/audit_report_numbers.py.bak-patch98")

OLD_TOKEN = '''CITATION = re.compile(r"arxiv[:\\s]*\\S+|\\b\\d+\\.\\d+\\.\\d+\\b", re.IGNORECASE)'''
NEW_TOKEN = OLD_TOKEN + '''
# An unfilled template field. Deliberately not the same kind of check as TOKEN:
# a placeholder has no digits, so no comparison against a record can find it, and
# the sentence around it reads as a complete statement about a quantity. Written in
# this file without a literal example, because this file's own text is scanned by
# nothing -- but the documents it reads are prose, and a reader who copies an
# example into one would otherwise reintroduce the defect from the fix.
PLACEHOLDER = re.compile(r"\\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\\}")'''
if a.count(OLD_TOKEN) != 1:
    sys.exit(f"audit anchor for TOKEN/CITATION appears {a.count(OLD_TOKEN)} times; "
             f"nothing written")
a = a.replace(OLD_TOKEN, NEW_TOKEN, 1)

OLD_SCAN = '''        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        seen, missing = set(), defaultdict(list)
        for i, line in enumerate(lines, 1):
            for tok in TOKEN.findall(CITATION.sub(" ", line)):'''
NEW_SCAN = '''        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        seen, missing = set(), defaultdict(list)
        for i, line in enumerate(lines, 1):
            for ph in PLACEHOLDER.findall(line):
                placeholders.append((doc.name, i, ph))
            for tok in TOKEN.findall(CITATION.sub(" ", line)):'''
if a.count(OLD_SCAN) != 1:
    sys.exit(f"audit anchor for the scan loop appears {a.count(OLD_SCAN)} times; "
             f"nothing written")
a = a.replace(OLD_SCAN, NEW_SCAN, 1)

OLD_INIT = '''    unreadable, report = [], []'''
NEW_INIT = '''    unreadable, report, placeholders = [], [], []'''
if a.count(OLD_INIT) != 1:
    sys.exit("audit anchor for the report list; nothing written")
a = a.replace(OLD_INIT, NEW_INIT, 1)

OLD_SUM = '''    print("\\n=== summary ===")'''
NEW_SUM = '''    if placeholders:
        print("\\n=== unfilled template fields ===")
        print("  A field here is not a wrong number -- it is no number, in a sentence")
        print("  that reads as though it had one. There is no legitimate prose form")
        print("  that looks like one, so this is a failure, not a list to review.")
        for name, lineno, ph in placeholders:
            print(f"    {name}:{lineno}  {ph}")
    print("\\n=== summary ===")'''
if a.count(OLD_SUM) != 1:
    sys.exit("audit anchor for the summary; nothing written")
a = a.replace(OLD_SUM, NEW_SUM, 1)

OLD_TAIL = '''    print(f"  untraceable       : {total_missing}")


if __name__ == "__main__":
    main()'''
NEW_TAIL = '''    print(f"  untraceable       : {total_missing}")
    print(f"  unfilled fields   : {len(placeholders)}")
    if placeholders:
        sys.exit(f"{len(placeholders)} unfilled template field(s) in the documents; "
                 "an unfilled field is a missing number, not a small one")


if __name__ == "__main__":
    main()'''
if a.count(OLD_TAIL) != 1:
    sys.exit("audit anchor for the exit block; nothing written")
a = a.replace(OLD_TAIL, NEW_TAIL, 1)

OLD_DOC = '''Exit status is 0 even when untraceable numbers are found: prose legitimately
contains derived arithmetic ("+0.108", "2.8x"), so the output is a list to read,
not a gate to trip.
"""'''
NEW_DOC = '''Exit status is 0 when untraceable numbers are found: prose legitimately contains
derived arithmetic ("+0.108", "2.8x"), so that output is a list to read, not a gate
to trip.

It is NOT 0 when a template field is found. A field is not a number that failed to
match a record; it is a number that was never written, and the sentence around it
reads as complete. No legitimate prose form looks like a template field, so a hit
fails the run -- which also stops tools/sync_counts.py, since it refuses to rewrite
the document's counts when the audit fails.
"""'''
if a.count(OLD_DOC) != 1:
    sys.exit("audit anchor for the docstring; nothing written")
a = a.replace(OLD_DOC, NEW_DOC, 1)

# ------------------------------------------------------------------ 2. the record
s = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch98")

OLD9 = '''- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 546 个指标数字，
  **无法回溯到任何记录的：0 个**；其中 5 个是已记录数字之间的差值（算术量）。'''
NEW9 = '''- 数字溯源核对 `tools/audit_report_numbers.py`：三份文档共 546 个指标数字，
  **无法回溯到任何记录的：0 个**；其中 5 个是已记录数字之间的差值（算术量）。
  该工具同时扫描**未替换的模板字段**（花括号后紧跟标识符的那种）—— 这类东西不是
  「算错了的数字」，而是**根本没有数字**，而句子仍然读得通（第 8 节第 14 条）。
  散文里不存在合法的这种形态，所以它不是「待复核清单」而是失败：一旦命中，
  该工具**非零退出**，`tools/sync_counts.py` 随之拒绝改写文档计数。'''
if s.count(OLD9) != 1:
    sys.exit(f"section 9 audit bullet anchor appears {s.count(OLD9)} times; nothing written")
s = s.replace(OLD9, NEW9, 1)

ANCHOR = "## 9. 可复现性核对"
ITEM14 = '''14. **两句带着未替换的模板字段发出去了 —— 不是数字错了，是根本没有数字。**
    5.2.3 与 5.2.5 各有一句写成模板，花括号里的字段名从未替换：读者看到的是字段名，
    不是数值。这一条与前 13 条的机制都不同：前 13 条里，数字要么本身错了，要么正确
    但关于一个已经不存在的 run 集合；**这里没有任何数字**，而且句子在排版上完整 ——
    「Δ准确率」后跟一个字段名，语法上是一句关于某个量的话，不试读数值看不出它没有
    被填过。它和 5.2.2 里 `round(p, 10)` 把 2.8e-26 写成 0 是同一种失败：
    **产物在未核对时与正确产物无法区分**，只是那次丢掉的是值、这次丢掉的是整个数。
    已由 `patch96_placeholders.py` 从 `paired_significance.json` 填入 —— 与 5.2.6
    引用的是同一条记录，所以两节不可能互相矛盾。
    处理方式与前几条一致，但落点不同：这次的核对**不能**放进数字溯源里，因为溯源靠
    比对数字，而字段没有数字可对。所以它作为一次独立的形态扫描放进
    `tools/audit_report_numbers.py`，并且是全项目唯一一处**命中即非零退出**的核对 ——
    散文里不存在合法的这种形态，因此不需要人读一遍再判断。

'''
if s.count(ANCHOR) != 1:
    sys.exit(f"section 9 anchor appears {s.count(ANCHOR)} times; nothing written")
s = s.replace(ANCHOR, ITEM14 + ANCHOR, 1)

if s.count("**") % 2:
    sys.exit("bold markers are odd; nothing written")

AUDIT.write_text(a, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(AUDIT)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/audit_report_numbers.py.bak-patch98", AUDIT)
    sys.exit(f"the patched audit does not compile, restored:\n{r.stderr}")
DOC.write_text(s, encoding="utf-8")
print(f"audit: placeholder gate added; LIMITATIONS.md: item 14 + section 9 sentence "
      f"({len(s.splitlines())} lines)")
