"""Render 5.2.7 offline, then run the third-draw patch against it -- twice.

patch95_seed3.py edits what patch94_seed.py writes, and patch94 only runs during a
landing. Its anchors are therefore text that does not exist on disk yet, which is the
one thing a patch should never be trusted about: the two rewrites it makes (5.2.7's
closing caveat, and 5.2.2's qualifier) are matched by regex, and a regex written from
a reading of the source rather than from the artefact is how a landing stops at 22:00
with a message about an anchor.

So 5.2.7 is rendered here from patch94 itself -- run against a COPY of the document,
with its record path redirected to the fixture built by sv_three_draw_fixture.py -- and
patch95 is then run against that copy, with WAFER_DOC so it cannot touch the real one.
Three things are checked: it writes, it is idempotent, and each branch it can take is
taken at least once (the range above the threshold while the pair is below; the range
below; and 5.2.2's amendment only in the first case).

Nothing under the project is written by this script.
"""
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
T = pathlib.Path("/tmp/svtest")
PY = sys.executable
SVEREC = T / "outputs/reports/seed_variance.json"
COPY = pathlib.Path("/tmp/doc_5.2.7.md")
# The patch under test is the production file, so the copy that reads the fixture is a
# redirection of it rather than a second version: a test run against a hand-edited copy
# proves something about the copy.
P95_SRC = ROOT / "tools/patch95_seed3.py"
P95 = T / "patch95_tmp.py"
_p95 = P95_SRC.read_text(encoding="utf-8")
for _old, _new, _what in ((f'SV = REP / "seed_variance.json"',
                           f'SV = pathlib.Path("{SVEREC}")', "the record path"),
                          ('REP = ROOT / "outputs/reports"',
                           f'REP = pathlib.Path("{T / "outputs/reports"}")',
                           "the reports directory")):
    if _p95.count(_old) != 1:
        sys.exit(f"patch95's {_what} is not where this script expects it; nothing run")
    _p95 = _p95.replace(_old, _new, 1)
P95.write_text(_p95, encoding="utf-8")

# ------------------------------------------------- 1. a clean three-draw fixture record
r = subprocess.run([PY, str(T / "sv.py")], capture_output=True, text=True, cwd=str(T))
if r.returncode != 0:
    sys.exit(f"the fixture tool exited {r.returncode}:\n{r.stdout[-800:]}{r.stderr[-400:]}")
rec = json.loads(SVEREC.read_text(encoding="utf-8"))
sp = rec["seed_spread"]
print(f"fixture record: {sp['n_draws']} draws, range {sp['range']:.4f} vs threshold "
      f"{sp['threshold_half_lr_effect']:.4f}, triggered={sp['rule_triggered']} "
      f"(pair={sp['rule_triggered_by_the_pair']}, agree={sp['rules_agree']})")

# ------------------------------------- 2. render 5.2.7 from patch94 onto a document copy
src = (ROOT / "tools/patch94_seed.py").read_text(encoding="utf-8")
# patch94 imports `from pathlib import Path` and nothing else, so the redirections have
# to be written in that module's own vocabulary.
for old, new, what in ((f'SV = REP / "seed_variance.json"', f'SV = Path("{SVEREC}")',
                        "the record path"),
                       ('DOC = ROOT / "LIMITATIONS.md"', f'DOC = Path("{COPY}")',
                        "the document path"),
                       ('REP = ROOT / "outputs/reports"',
                        f'REP = Path("{T / "outputs/reports"}")',
                        "the reports directory (the per-run reports it quotes)")):
    if src.count(old) != 1:
        sys.exit(f"patch94's {what} is not where this script expects it; nothing run")
    src = src.replace(old, new, 1)
(T / "patch94_tmp.py").write_text(src, encoding="utf-8")
shutil.copy2(ROOT / "LIMITATIONS.md", COPY)
r = subprocess.run([PY, str(T / "patch94_tmp.py")], capture_output=True, text=True,
                   cwd=str(ROOT))
print(f"\n--- patch94 (into {COPY}): exit {r.returncode} ---")
for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-6:]:
    print("  " + ln[:170])
doc = COPY.read_text(encoding="utf-8")
if "#### 5.2.7" not in doc:
    sys.exit("5.2.7 was not rendered; the rest of this check cannot run")

i = doc.find("**两次种子给不出方差。**")
print("\n5.2.7's closing paragraph, as patch94 writes it:")
print("  " + repr(doc[i:i + 300])[:460] if i >= 0 else "  NOT FOUND")


def run95(label, doc_text=None):
    if doc_text is not None:
        COPY.write_text(doc_text, encoding="utf-8")
    env = dict(**__import__("os").environ, WAFER_DOC=str(COPY))
    r = subprocess.run([PY, str(P95)], capture_output=True, text=True, cwd=str(ROOT), env=env)
    print(f"\n--- patch95 [{label}]: exit {r.returncode} ---")
    for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-8:]:
        print("  " + ln[:170])
    return r


# ---------------------------------------------------- 3. the case the fixture is in
run95("range above the threshold, pair below")
got = COPY.read_text(encoding="utf-8")
if "#### 5.2.8" not in got:
    sys.exit("5.2.8 was not written")
if "本项目没跑" in got.split("#### 5.2.7")[1].split("#### 5.2.8")[0]:
    sys.exit("5.2.7 still says the extra seeds were never run")
j = got.find("#### 5.2.8")
k = got.find("## 6. 评测与指标解释")
print("\n5.2.8 as written (first 700 chars):")
print("  " + got[j:j + 700].replace("\n", "\n  "))
print("\n5.2.2's qualifier tail:")
m = got.find("第三个抽样把这条限定加宽了")
print("  " + (got[m - 120:m + 200].replace("\n", "\n  ") if m >= 0 else "NOT AMENDED"))
if not sp["rules_agree"] and sp["rule_triggered"] and m < 0:
    sys.exit("the range crossed the threshold but 5.2.2 was not amended")
if k < j:
    sys.exit("5.2.8 was written after section 6")

# ------------------------------------------------------------- 4. idempotent?
run95("second run on the same document")
if "#### 5.2.8" in COPY.read_text(encoding="utf-8") and COPY.read_text(
        encoding="utf-8").count("#### 5.2.8") != 1:
    sys.exit("a second run wrote a second 5.2.8")

# ------------------------------------------------- 5. the other branch: below threshold
rec2 = json.loads(SVEREC.read_text(encoding="utf-8"))
rec2["seed_spread"]["rule_triggered"] = False
rec2["seed_spread"]["rule_triggered_by_the_pair"] = False
rec2["seed_spread"]["rules_agree"] = True
rec2["seed_spread"]["reading"] = ("over 3 draws the range is below half the "
                                  "learning-rate effect")
SVEREC.write_text(json.dumps(rec2, ensure_ascii=False, indent=2), encoding="utf-8")
shutil.copy2(ROOT / "LIMITATIONS.md", COPY)
r = subprocess.run([PY, str(T / "patch94_tmp.py")], capture_output=True, text=True,
                   cwd=str(ROOT))
if r.returncode != 0:
    sys.exit(f"the fixture could not render a second 5.2.7: {r.stdout[-400:]}")
run95("range below the threshold")
got = COPY.read_text(encoding="utf-8")
if "第三个抽样把这条限定加宽了" in got:
    sys.exit("5.2.2 was amended although the range did not cross the threshold")
if "**未达阈值**" not in got:
    sys.exit("the below-threshold branch did not write its own reading")

print("\n5.2.8 applies in both branches, is idempotent, and amends 5.2.2 only when the "
      "range moves the reading; the real LIMITATIONS.md was not touched")
