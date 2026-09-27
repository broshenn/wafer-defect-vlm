"""Close two blind spots in the tool that certifies the document.

This is the checker's own version of the defect the document exists to record. Every
number it prints is correct; the question is whether it ever asked the question the
sentence beside the number answers. Twice it did not.

1. `CLASSES = ("Donut", "none", "Edge_Ring", "Scratch")` -- four of the nine classes
   every report carries. It is the population the "恰好只有两个有类别归零" sentence is
   checked against, and it is frozen at the classes that had been zeroed when the line
   was written. `Center`, `Edge_Loc`, `Loc`, `Near_full` and `Random` are not looked at.
   Today none of the five is zeroed, so the sentence is true and the checker is green --
   the two agree by luck, not by inspection. Run 41 is another G=32 run, and the
   nearest-to-zero classes anywhere in the table are `Loc` (0.274) and `Near_full`
   (0.316), both on a GSPO run. If it zeroes one, the sentence goes false and this tool
   keeps saying ok. A certificate that cannot show it asked is the thing section 8 item
   9 is about, and it is worse here than in the prose: the prose is read by a person.

   So the class list is derived from the reports instead of declared. The classes are
   also compared between reports first -- a report that dropped a class would change
   every "keeps every class" answer below while looking like a smaller table.

2. "5.2.5: how many lr 1e-5 runs keep both classes" computed
   `len([n for n in rl if n.endswith("lr1e-5")])` -- the size of the set, not the number
   in it that keeps both classes. The item's name and the sentence's words ("两个类别
   都在") are about the property. While no lr 1e-5 run zeroes a class the two numbers
   are equal, so it passed; the moment one does, a numeral that satisfies this tool is
   a sentence that says the opposite of the record. It is compared against the number
   that actually keeps every class now.

Neither change moves any verdict on the document as it stands: the zeroed set is still
{ GSPO(G=8) lr5e-5: none, GSPO(G=32) lr5e-5: Donut } and the intact lr 1e-5 runs are
still all three. Both are checked by re-running the tool afterwards and requiring it to
still exit 0.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch102"

# ---------------------------------------------------------------- 1. the class list
OLD_CLASSES = 'CLASSES = ("Donut", "none", "Edge_Ring", "Scratch")\n'

NEW_CLASSES = '''# The class list is derived, not declared -- see the block after the records are read.
# A hardcoded subset only asks about the classes that had been seen to zero, which is
# the one thing a run that lands later is free to change without the sentence beside it
# moving. The four names that used to be here (Donut, none, Edge_Ring, Scratch) are the
# classes the sentences happen to name; they are a subset of the answer, not the answer.

'''

OLD_AFTER_RL = '''rl = [n for n in runs if n != "SFT"]              # the RL runs
print(f"records on disk: {len(runs)} runs ({len(rl)} RL + SFT); "
      f"absent: {sorted(set(RUNS) - set(runs)) or 'none'}")
'''

NEW_AFTER_RL = OLD_AFTER_RL + '''
# Every class any report carries, sorted. Derived from the records so that a class no
# run had zeroed when this tool was written is still looked at by the zeroing sentences
# below -- those sentences say "a class", not "one of four classes".
CLASSES = tuple(sorted({c for r in present.values() for c in r["per_class"]}))
'''

OLD_TABLES = '''cmp("the reward table's columns match the RL runs on disk",
    sorted(r_cols), sorted(rl),
    why="this table carries no SFT column -- it is the runs with a reward signal")
'''

NEW_TABLES = OLD_TABLES + '''
# Compared before the class sets are used: a report that lost a class would make every
# "does this run keep every class" answer below refer to a smaller population, and the
# only visible symptom would be a class missing from a table nobody re-counted.
_class_sets = {n: tuple(sorted(v["per_class"])) for n, v in present.items()}
_variants = sorted(set(_class_sets.values()))
cmp("every report carries the same class set", _variants, _variants[:1],
    why="two reports that disagree on the classes cannot be compared per class; and a "
        "class that is simply absent reads as a class with no score, which the intact "
        "checks below would count as a lost class")
'''

# --------------------------------------------------- 2. the lr 1e-5 sentence's number
OLD_LOW = '''    low = [n for n in rl if n.endswith("lr1e-5")]
    cmp("5.2.5: the number of lr 1e-5 RL runs", cn2int(g.group(1)), len(low),
        why="the sentence also names the values of those runs, so a third one is a "
            "rewrite, not a count bump")
    print(f"         they are: {'、'.join(low)}")'''

NEW_LOW = '''    low = [n for n in rl if n.endswith("lr1e-5")]
    # The sentence asserts a property of these runs -- "两个类别都在" -- so the number it
    # is checked against has to be the number that keeps every class, not the size of the
    # set. While no lr 1e-5 run zeroes a class the two are equal and this passed either
    # way; once one does, comparing against the size would certify a sentence saying the
    # opposite. Which classes count as "every" is CLASSES, derived above from the reports.
    intact_low = [n for n in low
                  if all(present[n]["per_class"].get(c, 0.0) > 0.0 for c in CLASSES)]
    cmp("5.2.5: the number of lr 1e-5 runs that keep every class",
        cn2int(g.group(1)), len(intact_low),
        why="the sentence also names the values of those runs, so a third one is a "
            "rewrite, not a count bump; and it claims they keep both classes, so a run "
            "in the set that zeroes one is a rewrite too")
    if len(intact_low) == len(low):
        print(f"         they are: {'、'.join(low)}")
    else:
        print(f"         they are: {'、'.join(low)} -- of which keeping every class: "
              f"{'、'.join(intact_low) or 'none'}")'''

s = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)

for old, new, what in ((OLD_CLASSES, NEW_CLASSES, "the CLASSES definition"),
                       (OLD_AFTER_RL, NEW_AFTER_RL, "the run-set block"),
                       (OLD_TABLES, NEW_TABLES, "the table-column checks"),
                       (OLD_LOW, NEW_LOW, "the lr 1e-5 count")):
    if s.count(old) != 1:
        sys.exit(f"{what} appears {s.count(old)} times, expected 1; nothing written")
    s = s.replace(old, new, 1)

# The derivation has to be reached before it is used, and it must come after `present`.
_i_class = s.find("CLASSES = tuple(sorted(")
_i_present = s.find("present = {n: v for n, v in records.items()")
_i_use = s.find("for c in CLASSES")
if not (_i_present < _i_class < _i_use):
    sys.exit(f"the derived CLASSES is not between the records and their first use "
             f"(present@{_i_present}, class@{_i_class}, use@{_i_use}); nothing written")
if s.count("CLASSES = ") != 1:
    sys.exit(f"CLASSES is assigned {s.count('CLASSES = ')} times; nothing written")

CHECKER.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("checker patched: the class list is derived from the reports (all of them, not "
      "four), and the lr 1e-5 sentence is checked against the runs that keep every "
      "class rather than the size of the set.")

# Both changes are tightenings, so the document as it stands must still be green. A run
# that now reports a stale item means the change disagrees with the document, and the
# change is the thing to suspect -- restore rather than leave a checker that condemns
# prose it used to accept.
print("\n--- re-running the patched checker (read-only) ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
tail = r.stdout.strip().splitlines()
for ln in tail[-14:]:
    print("  " + ln)
if r.returncode == 0:
    print("\nthe document is still green under the stricter checker; both false "
          "negatives are closed")
elif r.returncode == 2:
    print(f"\nONLY COUNTS MOVED (exit 2) -- the document has a stale count for another "
          f"reason; the scope fixes themselves are fine")
else:
    shutil.copy2(BAK, CHECKER)
    print(f"\nthe stricter checker flags a fact in the current document (exit "
          f"{r.returncode}); the checker has been RESTORED. The changes were not "
          f"verified and must not be left in place.")
