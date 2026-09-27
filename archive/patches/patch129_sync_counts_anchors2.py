"""The last two anchors in `tools/sync_counts.py` that no longer match the document.

patch128 repaired the first dead anchor (`8.9: idle-step total`), which was the one that
made the tool exit before writing anything. An ast probe over every pattern in the file
then named the other two, and both drifted the same way the first one did: the sentence
was reworded, the anchor kept insisting on the old wording, and the tool's own rule --
"a missing anchor means the sentence was reworded and this script would otherwise edit
nothing while reporting success" -- turns a rewording into a hard exit, correctly.

  1. `9: per-value check count`. The document reads `347个空转步上的逐值核对`; the
     anchor requires `347 个`. Same one-space drift as 8.9, same fix, same reason: the
     numeral is the fact, the spacing is typography.
  2. `5.2.5: the count the four bullets below must be re-checked against`. The
     document now reads

         （此表的列数是随时间增长的：写此句时为八列。在任何新 run 落地后，下面四句里的
         那个 run 数都必须重核而不是沿用 —— ...）

     which is a *better* sentence than the form the anchor was written for
     (`下面四句里的「N个 run」都必须重核`): it says the count has to be re-derived
     rather than re-typed. It is also, for that exact reason, unreadable by the tool
     whose job is to re-derive it. So the anchor accepts both wordings and re-emits the
     form it owns, numeral included -- `下面四句里的「九个 run」都必须重核而不是沿用` --
     keeping the trailing clause that both forms share.

Both rewrites are checked by re-running `--check` and by hashing the document before and
after: `--check` must report edits and write nothing.
"""
import hashlib
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
SYNC = ROOT / "tools/sync_counts.py"
DOC = ROOT / "LIMITATIONS.md"

EDITS = [
    # ---- 1. `347个` vs `347 个`
    (
        r'''    sub(r"[一二三四五六七八九十]+个( run 全部 )\d+( 个空转步上的逐值核对)",''',
        r'''    # `\d+\s*`: the same drift as 8.9, one line up. The numeral is the fact this
    # sentence carries; whether the document writes `347 个` or `347个` is typography,
    # and requiring one of the two made the tool exit without writing any of its
    # counters -- the failure mode patch128 was about, in its second instance.
    sub(r"[一二三四五六七八九十]+个( run 全部 )\d+\s*(个空转步上的逐值核对)",''',
    ),
    # ---- 2. `那个 run 数都必须重核而不是沿用` vs `「N个 run」都必须重核`
    (
        r'''    sub(r"(下面四句里的\s*\n?\s*「)[一二三四五六七八九十]+(个 run」都必须重核)",
        f"下面四句里的\n「{numeral(cols)}个 run」都必须重核",
        "5.2.5: the count the four bullets below must be re-checked against")''',
        r'''    # Both wordings, one sentence. The document drifted to `下面四句里的 / 那个 run 数
    # 都必须重核而不是沿用`, which is the *better* statement of the rule -- re-derive the
    # count, do not carry it over -- and is exactly why the tool that re-derives it must
    # still be able to read it. The anchor accepts either, and re-emits the numeral it
    # computed, keeping the trailing clause both wordings share.
    sub(r"下面四句里的\s*\n?\s*(?:「[一二三四五六七八九十]+个 run」|那个 run 数)"
        r"\s*都必须重核(?:而不是沿用)?",
        f"下面四句里的\n「{numeral(cols)}个 run」都必须重核而不是沿用",
        "5.2.5: the count the four bullets below must be re-checked against")''',
    ),
]

s = SYNC.read_text(encoding="utf-8")
shutil.copy2(SYNC, "/tmp/sync_counts.py.bak-patch129")
for i, (old, new) in enumerate(EDITS, 1):
    if s.count(old) != 1:
        sys.exit(f"edit {i}: the anchor matches {s.count(old)} times; nothing written")
    s = s.replace(old, new, 1)
SYNC.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(SYNC)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/sync_counts.py.bak-patch129", SYNC)
    sys.exit(f"sync_counts does not compile; restored:\n{r.stderr}")
print(f"sync_counts.py: {len(EDITS)} anchors loosened (spacing in one, wording in one).")


def digest(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


# `--check` must report what it would change and write nothing -- asserted, not assumed.
for label in ("check", "check again"):
    before, before_sync = digest(DOC), digest(SYNC)
    r = subprocess.run([sys.executable, str(SYNC), "--check"], capture_output=True,
                       text=True, cwd=str(ROOT))
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    print(f"\n--- sync_counts --check ({label}): exit {r.returncode} ---")
    for ln in out.splitlines():
        print("  " + ln.strip()[:170])
    if r.returncode == 0:
        print(f"  [check] document unchanged by --check: "
              f"{'yes' if digest(DOC) == before else 'NO -- it wrote!'}"
              f"  (tool unchanged: {'yes' if digest(SYNC) == before_sync else 'NO'})")
