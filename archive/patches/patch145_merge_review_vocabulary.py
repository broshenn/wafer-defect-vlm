"""A reviewer's typo was reported as a benchmark defect.

merge_review.py compares each reviewer's `defect_class` to the core file's `failure_type`
and sorts every sample into three outcomes. Its own docstring gives the meaning of the
second one: "the reviewers agree with each other but not with gold -> the automatic label
is wrong, which is a benchmark defect rather than a reviewer problem". Nothing checks that
the label is a label: `normalise` strips whitespace and that is all. So a reviewer who
writes `Edgeloc` for `Edge_Loc` puts that sample in `gold_mismatch` -- the report then
tells the reader the benchmark's gold is wrong at that sample, which is a claim about the
data drawn from a typo. A label written in the wrong case (`center`) becomes a
`reviewer_disagreement` instead, i.e. "a genuinely ambiguous sample".

The pre-flight run measured it on two synthetic sheets in /tmp: eight out-of-vocabulary
labels turned 13 gold mismatches into 17 and 8 disagreements into 12, moved the reported
agreement from 0.9654 to 0.9498, and the tool exited 0. A merge whose numbers are that
contaminated is not distinguishable from a clean one.

The fix is a vocabulary check, and the vocabulary is not a new list: it is the set of
`failure_type` values in the core file being merged, which is exactly the space the
README tells the reviewers to choose from. A label outside it is an input error, so it is
counted separately, kept out of the agreement statistics (it is not evidence about the
benchmark or about the other reviewer), and the tool exits 3 -- the report is still
printed, because it is what the reader needs in order to fix the sheets.

cohen_kappa and normalise are untouched: the project's five unit tests cover those two
functions, and they are run after this patch.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/merge_review.py"

OLD_DOC = """- the reviewers disagree with each other -> genuinely ambiguous sample.

Only confirmed samples are marked approved, which is what ``curate``/``freeze``
requires before the benchmark can stop being a draft.
\"\"\""""

NEW_DOC = """- the reviewers disagree with each other -> genuinely ambiguous sample;
- a reviewer wrote something that is not one of the benchmark's classes -> an input
  error, reported separately and *not* read as evidence about the gold label.

The fourth outcome exists because the first three are claims about the data. A typo
in a sheet would otherwise be reported as a benchmark defect (agreement with gold) or
as an ambiguous sample (disagreement), and a run contaminated that way exited 0, so it
could not be told from a clean one. The vocabulary is the core file's own set of
``failure_type`` values -- the space the review README lists -- so there is no second
list to go stale.

Only confirmed samples are marked approved, which is what ``curate``/``freeze``
requires before the benchmark can stop being a draft.
\"\"\""""

OLD_LOOP = """    pairs: list[tuple[str, str]] = []
    outcomes = Counter()
    confirmed, gold_mismatch, reviewer_disagreement, unlabelled = [], [], [], []

    for row in core:
        sid = row["sample_id"]
        a, b = sheet_a.get(sid, {}), sheet_b.get(sid, {})
        va, vb = normalise(a.get("defect_class")), normalise(b.get("defect_class"))
        if not va or not vb:
            unlabelled.append(sid)
            outcomes["missing_review"] += 1
            continue
        pairs.append((va, vb))"""

NEW_LOOP = """    # The benchmark's own label space, read from the file being merged rather than
    # written down a second time: a class the gold labels do not use is not a class a
    # reviewer may return.
    vocabulary = {row["failure_type"] for row in core}

    pairs: list[tuple[str, str]] = []
    outcomes = Counter()
    confirmed, gold_mismatch, reviewer_disagreement, unlabelled = [], [], [], []
    out_of_vocabulary: list[dict[str, object]] = []

    for row in core:
        sid = row["sample_id"]
        a, b = sheet_a.get(sid, {}), sheet_b.get(sid, {})
        va, vb = normalise(a.get("defect_class")), normalise(b.get("defect_class"))
        if not va or not vb:
            unlabelled.append(sid)
            outcomes["missing_review"] += 1
            continue
        unknown = [v for v in (va, vb) if v not in vocabulary]
        if unknown:
            # Not evidence, so not in `pairs`: an out-of-vocabulary label says something
            # about the sheet, not about the sample. Counting it either way would put a
            # claim about the data into the report on the strength of a spelling.
            out_of_vocabulary.append({"sample_id": sid, "reviewer_a": va,
                                      "reviewer_b": vb, "unknown": unknown})
            outcomes["out_of_vocabulary"] += 1
            continue
        pairs.append((va, vb))"""

OLD_REPORT = """        "unlabelled_samples": unlabelled[:50],
        "note": "gold_match is the benchmark's own accuracy, not a model score",
    }"""

NEW_REPORT = """        "unlabelled_samples": unlabelled[:50],
        "out_of_vocabulary_samples": out_of_vocabulary[:50],
        "vocabulary": sorted(vocabulary),
        "note": "gold_match is the benchmark's own accuracy, not a model score. An "
                "out_of_vocabulary label is a sheet error and is excluded from every "
                "agreement statistic above.",
    }"""

OLD_TAIL = """    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0"""

NEW_TAIL = """    print(json.dumps(report, ensure_ascii=False, indent=2))
    if out_of_vocabulary:
        print(f"\\n{len(out_of_vocabulary)} sample(s) carry a label outside the "
              f"benchmark's vocabulary ({len(vocabulary)} classes); they are listed in "
              f"out_of_vocabulary_samples and are excluded from the statistics. The "
              f"report above is NOT a clean merge.", file=sys.stderr)
        return 3
    return 0"""

s = TOOL.read_text(encoding="utf-8")
if "out_of_vocabulary" in s:
    sys.exit("merge_review.py already checks the vocabulary; nothing written")
shutil.copy2(TOOL, "/tmp/merge_review.py.bak-patch145")

for what, old, new in (("the docstring", OLD_DOC, NEW_DOC),
                       ("the outcome loop", OLD_LOOP, NEW_LOOP),
                       ("the report", OLD_REPORT, NEW_REPORT),
                       ("the return", OLD_TAIL, NEW_TAIL)):
    if s.count(old) != 1:
        shutil.copy2("/tmp/merge_review.py.bak-patch145", TOOL)
        sys.exit(f"{what} matches {s.count(old)} times; nothing written")
    s = s.replace(old, new, 1)
TOOL.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/merge_review.py.bak-patch145", TOOL)
    sys.exit(f"the patched merge_review.py does not compile, restored:\n{r.stderr}")
print("merge_review.py: out-of-vocabulary labels are an input error, not evidence")

# ------------------------------------------- the project's own tests still pass
t = subprocess.run([sys.executable, "-m", "pytest",
                    str(ROOT / "projects/wafer-defect-vlm/tests/test_merge_review.py"),
                    "-q"], capture_output=True, text=True, cwd=str(ROOT))
print(f"\npytest test_merge_review.py: exit {t.returncode}")
print("  " + ((t.stdout or t.stderr).strip().splitlines() or ["(no output)"])[-1][:170])
if t.returncode != 0:
    sys.exit("the project's own tests fail after the patch")
