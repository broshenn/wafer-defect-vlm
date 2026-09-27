"""Exercise the human-review merge with synthetic sheets, and never touch the real ones.

The merge's decisive branches have never run: agreement.json on disk was produced from two
sheets with every label cell empty, so `pairs` was empty, cohen_kappa returned None before
its division, and neither `confirmed` nor `gold_mismatch` was ever incremented. The branch
that runs once the two humans are done is therefore the least exercised code on the path
to freezing the benchmark -- and it is the branch whose output decides which gold labels
are questions and which samples are approved.

So: two synthetic sheets are built in /tmp from the real core file, deliberately containing
all four outcomes plus two ways of writing a label that is not in the benchmark's
vocabulary (a missing underscore, and the right word in the wrong case). The tool runs
against a /tmp copy of core; the real review/ tree and core.jsonl are checksummed before
and after and must be identical.

The expected counts are recomputed here, independently of the tool: this script counts the
synthetic sheets with its own implementation of the documented rule, and compares. A test
that reads the tool's number back out of the tool proves nothing.

Run before and after the vocabulary patch: the summary line names where each typo landed,
which is the whole point -- an out-of-vocabulary label must not be reported as a wrong
automatic label, nor as two reviewers disagreeing.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
PY = str(ROOT / "venvs/wafer/bin/python")
CORE = ROOT / "benchmarks/wafer_bench_v1/core.jsonl"
REVIEW = ROOT / "benchmarks/wafer_bench_v1/review"
MERGE = ROOT / "tools/merge_review.py"
T = pathlib.Path("/tmp/review_preflight")
T.mkdir(parents=True, exist_ok=True)

FIELDS = ["sample_id", "image_path", "defect_class", "radial_zone",
          "clock_direction", "confidence_0_100", "notes"]
MISSING_UNDERSCORE = "Edgeloc"          # a legal class, written wrong
WRONG_CASE = "center"                   # a legal class, written in another case


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


before = {str(p): md5(p) for p in [CORE, REVIEW / "reviewer_A.csv",
                                   REVIEW / "reviewer_B.csv"]}
review_listing = sorted(p.name for p in REVIEW.iterdir())

core = [json.loads(l) for l in CORE.read_text(encoding="utf-8").splitlines() if l.strip()]
vocabulary = sorted({r["failure_type"] for r in core})
print(f"core rows: {len(core)}   gold vocabulary: {len(vocabulary)} -> {vocabulary}")

# ------------------------------------------------------------------ the sheets
sheet_a, sheet_b, plan = [], [], {}
for i, row in enumerate(core):
    gold = row["failure_type"]
    other = [c for c in vocabulary if c != gold]
    a = b = gold
    kind = "confirmed"
    if i % 37 == 0:
        a = b = ""; kind = "missing_review"
    elif i % 41 == 0:
        b = ""; kind = "missing_review"
    elif i % 29 == 0:
        a, b = gold, other[i % len(other)]; kind = "reviewer_disagreement"
    elif i % 19 == 0:
        a = b = other[(i + 3) % len(other)]; kind = "gold_mismatch"
    elif i % 53 == 0:
        a = b = MISSING_UNDERSCORE; kind = "typo_both"
    elif i % 61 == 0:
        a, b = WRONG_CASE, gold; kind = "typo_one_side"
    plan[row["sample_id"]] = kind
    sheet_a.append(a)
    sheet_b.append(b)

for name, labels in (("reviewer_A_synthetic.csv", sheet_a),
                     ("reviewer_B_synthetic.csv", sheet_b)):
    with (T / name).open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(FIELDS) + "\n")
        for row, lab in zip(core, labels):
            cells = [row["sample_id"], row["image_path"], lab, "", "", "", ""]
            fh.write(",".join('"' + c.replace('"', '""') + '"' if "," in c else c
                              for c in cells) + "\n")
(T / "core_copy.jsonl").write_text(CORE.read_text(encoding="utf-8"), encoding="utf-8")

counts = {}
for kind in plan.values():
    counts[kind] = counts.get(kind, 0) + 1
print(f"synthetic sheets written: {counts}")

# ------------------------------------- the expected counts, recomputed here
labelled = [(plan[s], (a, b)) for s, a, b in
            zip([r["sample_id"] for r in core], sheet_a, sheet_b)
            if a and b]
legal_pairs = [(a, b) for _, (a, b) in labelled if a in vocabulary and b in vocabulary]
expected_agree = sum(a == b for a, b in legal_pairs) / len(legal_pairs)
illegal_pairs = [(a, b) for _, (a, b) in labelled
                 if a not in vocabulary or b not in vocabulary]
print(f"independently: {len(legal_pairs)} both-legal pairs, agreement "
      f"{expected_agree:.6f}, {len(illegal_pairs)} pair(s) with a label outside the "
      f"vocabulary")

# ------------------------------------------------------------------ run the tool
cmd = [PY, str(MERGE), "--a", str(T / "reviewer_A_synthetic.csv"),
       "--b", str(T / "reviewer_B_synthetic.csv"),
       "--core", str(T / "core_copy.jsonl"),
       "--output", str(T / "agreement_synthetic.json")]
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"\nmerge_review exit {r.returncode}")
if r.returncode not in (0, 3):
    print((r.stdout or "")[-500:] + (r.stderr or "")[-500:])
    sys.exit("the merge did not report; the pre-flight cannot judge anything")
report = json.loads(r.stdout)
print(json.dumps({k: v for k, v in report.items()
                  if not k.endswith("_samples")}, ensure_ascii=False, indent=2))
print("\nthe two typo rows, and where the report put them:")
for sid, kind in plan.items():
    if not kind.startswith("typo"):
        continue
    where = []
    for key in ("confirmed", "gold_mismatch", "reviewer_disagreement",
                "out_of_vocabulary"):
        entries = report.get(key + "_samples") or []
        # out_of_vocabulary_samples holds records, the other three hold sample ids;
        # comparing an id to a dict silently reports "not listed" for every row, which
        # is how this line read on its first run.
        ids = [e["sample_id"] if isinstance(e, dict) else e for e in entries]
        if sid in ids:
            where.append(key)
    print(f"  {sid}  {kind:14s} -> " + (", ".join(where) or "NOT LISTED ANYWHERE"))

# ------------------------------------------------- is the real tree untouched?
after = {str(p): md5(p) for p in [CORE, REVIEW / "reviewer_A.csv",
                                  REVIEW / "reviewer_B.csv"]}
same_listing = sorted(p.name for p in REVIEW.iterdir()) == review_listing
print(f"\nreal core.jsonl and both reviewer sheets unchanged: {before == after}")
print(f"review/ directory listing unchanged: {same_listing}")
if before != after or not same_listing:
    sys.exit("the pre-flight wrote into the benchmark tree; that must never happen")

# ------------------------------------------------------- does it agree with us?
ok = True
# reviewed_by_both must count the legal pairs only: an out-of-vocabulary label is not a
# review of the sample, it is a mistake in the sheet.
if report["reviewed_by_both"] != len(legal_pairs):
    print(f"FAIL: the tool counted {report['reviewed_by_both']} reviewed pairs; the "
          f"both-legal population is {len(legal_pairs)}")
    ok = False
agree = report.get("class_agreement_between_reviewers")
match = abs((agree or 0) - expected_agree) < 1e-9
print(f"agreement: tool {agree} vs independent {expected_agree:.6f} "
      f"-> {'match' if match else 'DIFFERENT'}")
ok = ok and match

# ------------------------------------------------- and the write path, on a copy
# --update-core rewrites every line of the core file. It is the write the freeze step
# depends on, and it has never been executed against a labelled sheet. Only the /tmp copy
# is written; the real core.jsonl is checksummed again afterwards.
UPDATE = T / "core_update.jsonl"
UPDATE.write_text(CORE.read_text(encoding="utf-8"), encoding="utf-8")
u = subprocess.run([PY, str(MERGE),
                    "--a", str(T / "reviewer_A_synthetic.csv"),
                    "--b", str(T / "reviewer_B_synthetic.csv"),
                    "--core", str(UPDATE), "--update-core"],
                   capture_output=True, text=True)
print(f"\n--update-core on a copy: exit {u.returncode}")
# 3 is the expected code here: this fixture deliberately contains eight typos, so the
# merge is not clean by construction. What is being checked is that the write path runs
# and that the approvals it writes do not depend on the typo rows.
if u.returncode not in (0, 3):
    print((u.stdout or "")[-300:] + (u.stderr or "")[-300:])
    sys.exit("the update path failed on a copy")
urep = json.loads(u.stdout)
before_rows = [json.loads(l) for l in CORE.read_text(encoding="utf-8").splitlines()
               if l.strip()]
after_rows = [json.loads(l) for l in UPDATE.read_text(encoding="utf-8").splitlines()
              if l.strip()]
changed = [(b, a) for b, a in zip(before_rows, after_rows) if b != a]
approved = [a for _, a in changed if a.get("agreement") == "approved"]
print(f"  lines changed: {len(changed)}   marked approved: {len(approved)}")
# Every change must be exactly the two fields, on a sample the report called confirmed.
NEW_FIELDS = ("annotator_count", "agreement")
bad = []
for b, a in changed:
    untouched = {k: v for k, v in a.items() if k not in NEW_FIELDS} == {
        k: v for k, v in b.items() if k not in NEW_FIELDS}
    if (set(a) - set(b) - set(NEW_FIELDS) or not untouched
            or a.get("annotator_count") != 2 or a.get("agreement") != "approved"):
        bad.append(a["sample_id"])
confirmed = urep["outcomes"].get("confirmed", 0)
print(f"  rows whose change is anything other than the two approved fields: {len(bad)}")
print(f"  marked approved {len(approved)} vs the report's confirmed count {confirmed}")
if bad or len(approved) != confirmed or not urep.get("core_updated"):
    print(f"  FAIL: {len(bad)} unexpected change(s), report "
          f"core_updated={urep.get('core_updated')}")
    ok = False

after = {str(p): md5(p) for p in [CORE, REVIEW / "reviewer_A.csv",
                                  REVIEW / "reviewer_B.csv"]}
print(f"  the real core.jsonl is still unchanged: {before == after}")
ok = ok and before == after
print(f"\nPRE-FLIGHT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
