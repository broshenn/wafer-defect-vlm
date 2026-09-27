"""`seed_variance.py` writes a `reading` that is true whichever way the run comes out.

The field is:

    "reading": "both seeds are the same side of SFT and neither is significantly
                below, or the null is not reproducible"

That is a disjunction of the two possible outcomes, so it cannot be false -- and for
the same reason it carries no information. It reads like a finding, sits where a
finding belongs, and is not one. This is the same defect the document records in
section 8 (item 14, the unsubstituted template fields): an artefact that cannot be
told from a complete one without checking. The difference is that a placeholder is
visibly a placeholder once you look, whereas this sentence looks like a conclusion at
a glance, which is worse.

The fix is the same one the rest of the project uses: compute the reading from the
values, and record the two booleans it is computed from so a reader can check the
sentence against the numbers rather than taking it on faith.

Applied to tools/seed_variance.py before queue 42 lands, so the record it writes is
the computed form from the start rather than being rewritten afterwards.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
SRC = ROOT / "tools/seed_variance.py"

OLD = '''    "replication_vs_sft": {
        "seed1": {"delta": round(acc1 - accs, 6), "p_exact_two_sided": pval(ps1)},
        "seed2": {"delta": round(d_vs_sft_2, 6), "p_exact_two_sided": pval(ps)},
        "reading": "both seeds are the same side of SFT and neither is "
                   "significantly below, or the null is not reproducible"},'''

NEW = '''    "replication_vs_sft": {
        "seed1": {"delta": round(acc1 - accs, 6), "p_exact_two_sided": pval(ps1)},
        "seed2": {"delta": round(d_vs_sft_2, 6), "p_exact_two_sided": pval(ps)},
        # Computed, not written by hand. This field used to be a disjunction of both
        # possible outcomes -- true however the run came out, and therefore worth
        # nothing to a reader, which is the failure section 8 item 14 records in the
        # document (an artefact indistinguishable from a complete one). The two
        # booleans the sentence is computed from are recorded beside it, so the
        # sentence can be checked against the numbers instead of believed.
        "same_side_of_sft": _same_side,
        "significantly_different_from_sft": _sig,
        "reading": (
            "not reproducible: the two seeds land on opposite sides of SFT "
            f"({acc1 - accs:+.4f} and {d_vs_sft_2:+.4f}), so the seed moves the "
            "comparison by more than the comparison resolves"
            if not _same_side else
            "not reproducible: the seeds agree on the side but one of them differs "
            f"from SFT significantly (p = {pval(ps1):.5f}, {pval(ps):.5f}), so the "
            "comparison is not stable across seeds"
            if _sig else
            "reproducible and null: both seeds are on the same side of SFT "
            f"({acc1 - accs:+.4f} and {d_vs_sft_2:+.4f}) and neither differs from "
            f"it significantly (p = {pval(ps1):.5f}, {pval(ps):.5f})"),
    },'''

PRE = '''out = {
    "question": "How large is run-to-run (seed) variation on this benchmark, and "'''
PRE_NEW = '''# The two facts the replication sentence is computed from.
_same_side = (acc1 - accs >= 0) == (d_vs_sft_2 >= 0)
_sig = bool(pval(ps1) < 0.05 or pval(ps) < 0.05)

out = {
    "question": "How large is run-to-run (seed) variation on this benchmark, and "'''

s = SRC.read_text(encoding="utf-8")
shutil.copy2(SRC, "/tmp/seed_variance.py.bak-patch97")

if s.count(OLD) != 1:
    sys.exit(f"the reading anchor appears {s.count(OLD)} times, expected 1; "
             f"nothing written")
s = s.replace(OLD, NEW, 1)

if s.count(PRE) != 1:
    sys.exit(f"the `out = {{` anchor appears {s.count(PRE)} times, expected 1; "
             f"nothing written")
s = s.replace(PRE, PRE_NEW, 1)

if "significantly below, or the null is not reproducible" in s:
    sys.exit("the old hand-written reading survived; nothing written")

SRC.write_text(s, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(SRC)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/seed_variance.py.bak-patch97", SRC)
    sys.exit(f"the patched tool does not compile, restored:\n{r.stderr}")

print("seed_variance.py: the replication reading is now computed from the two "
      "booleans, which are recorded beside it")
