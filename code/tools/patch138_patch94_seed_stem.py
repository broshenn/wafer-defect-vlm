"""patch94's seed parenthetical could never appear: the fallback hid the path bug.

`_seed(stem)` builds `REP / f"{stem}_train_result.json"`, and its two call sites pass
the *file name* rather than the tag:

    _S1 = _seed("grpo_train_result")
    _S2 = _seed("qwen35_9b_grpo_lr1e5_seed3408_train_result")

so it looked for `grpo_train_result_train_result.json` and
`qwen35_9b_grpo_lr1e5_seed3408_train_result_train_result.json`. Neither exists, both
lookups returned None, and the sentence the function exists to build --

    run 42 把 GRPO(G=4, lr 1e-5) 换种子（3407 → 3408）重跑

-- was written without its parenthetical in *every* case, including the case where both
records are on disk and both seeds are readable. The designed fallback ("drop the
literal rather than print one a reader cannot check") is correct; the defect is that a
wrong path reaches it, and a fallback that a path bug can reach is a fallback that
hides the bug. This is section 8 item 15's shape -- a mechanism that exists and never
ran -- found in the one patch the landing about to run is the last step of.

Found by rendering 5.2.7 offline against a copy of the document: the render printed
"seeds: (not in the records; omitted)" while `ls` showed both records on disk, which is
the only reason it was looked at. It would have gone unnoticed in production, because
the sentence it produces is grammatical and true -- just less checkable than intended.

The fix is the two stems, plus a line that says which record was unreadable when one
is, so the next occurrence of this shape is visible in the log instead of in the prose.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/patch94_seed.py"

OLD = '''_S1 = _seed("grpo_train_result")
_S2 = _seed("qwen35_9b_grpo_lr1e5_seed3408_train_result")
_seed_txt = f"（{_S1} → {_S2}）" if (_S1 is not None and _S2 is not None) else ""'''

NEW = '''# The stem is the tag, and `_seed` appends the suffix: passing the file's own name
# produced "<tag>_train_result_train_result.json", which never exists, so the
# parenthetical was dropped in every case -- silently, because dropping it is the
# designed fallback for "that record is not on disk". A fallback a path bug can reach
# hides the bug; the line below makes the next one visible in the log.
_S1 = _seed("grpo")
_S2 = _seed("qwen35_9b_grpo_lr1e5_seed3408")
_UNREADABLE = [name for name, s in
               (("grpo_train_result.json", _S1),
                ("qwen35_9b_grpo_lr1e5_seed3408_train_result.json", _S2))
               if s is None]
if _UNREADABLE:
    print("  seed parenthetical omitted: no `seed` in "
          + ", ".join(_UNREADABLE))
_seed_txt = f"（{_S1} → {_S2}）" if (_S1 is not None and _S2 is not None) else ""'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/patch94_seed.py.bak-patch138")
if s.count(OLD) != 1:
    sys.exit(f"the anchor matches {s.count(OLD)} times; nothing written")
TOOL.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/patch94_seed.py.bak-patch138", TOOL)
    sys.exit(f"the patched patch94_seed.py does not compile, restored:\n{r.stderr}")
print("patch94_seed.py: the seed stems are the tags, and an unreadable record is named")

# --------------------------------------------------------- and prove it on a copy
# The production document is not touched: patch94 is pointed at a copy, and at records
# it can already read. What is being checked is that the two stems resolve -- the
# parenthetical has to appear, and the omission line has to stay silent.
COPY = pathlib.Path("/tmp/doc_5.2.7b.md")
shutil.copy2(ROOT / "LIMITATIONS.md", COPY)
src = TOOL.read_text(encoding="utf-8")
for old, new, what in (('DOC = ROOT / "LIMITATIONS.md"', f'DOC = Path("{COPY}")',
                        "the document path"),
                       ('SV = REP / "seed_variance.json"',
                        'SV = Path("/tmp/svtest/outputs/reports/seed_variance.json")',
                        "the record path"),
                       ('REP = ROOT / "outputs/reports"',
                        'REP = Path("/tmp/svtest/outputs/reports")',
                        "the reports directory")):
    if src.count(old) != 1:
        sys.exit(f"patch94's {what} is not where this script expects it; nothing run")
    src = src.replace(old, new, 1)
TMP = pathlib.Path("/tmp/patch94_check.py")
TMP.write_text(src, encoding="utf-8")
r = subprocess.run([sys.executable, str(TMP)], capture_output=True, text=True, cwd=str(ROOT))
print(f"\n--- patch94 (into {COPY}): exit {r.returncode} ---")
for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-8:]:
    print("  " + ln[:170])
if r.returncode != 0:
    sys.exit("the redirect-and-run failed; see above")
print("\nthe rendered 5.2.7 design paragraph:")
doc = COPY.read_text(encoding="utf-8")
i = doc.find("run 42 把 GRPO(G=4, lr 1e-5) 换种子")
print("  " + doc[i:i + 150].replace("\n", " ") if i >= 0 else "  (not found)")
