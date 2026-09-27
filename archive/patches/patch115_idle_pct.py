"""The third list writes percentages, so the pattern built for the other two cannot read it.

`_PAIR_ITEM` reads `ALGO(G=n) lo → hi`, which is the shape of the KL pairs (1.1068 → 1.0435)
and of the completion lengths (115.48 → 97.92). The idle-step list is the same enumeration
of the same population in percentages --

    GSPO(G=8) 15.33% → 38.00%（`idle_delta` +0.2267）

-- and `([\\d.]+) → ` cannot cross the `%` between `15.33` and ` → `. So `findall` returned an
empty list and the item reported the sentence as enumerating nothing, while the count item
in the same branch read 三 and passed: a **green count beside an empty enumeration**, which
is the exact combination the second item exists to make impossible. It would have gone
unnoticed if the count and the enumeration were one item.

The lesson is the one this session keeps re-learning: a probe that finds nothing and says so
is worth more than one that finds nothing and says ok -- but the reason this one was caught
is that it reported an empty *set* against a three-member one rather than reporting a
population of zero that happened to match.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch115"

C_OLD = '''        _iitems = re.findall(_PAIR_ITEM + r"%（`idle_delta` ([−+]?[\\d.]+)）", _gip.group(2))
'''

C_NEW = '''        # Same enumeration, percentages rather than values: `_PAIR_ITEM` cannot cross
        # the `%` between the first number and the arrow, and a findall that matches
        # nothing returns an empty set, which is not the same as a document that
        # enumerates nothing -- the count item beside this one would still pass.
        _IPCT = (r"([A-Za-z]+)\\(G=(\\d+)\\) ([\\d.]+)% → ([\\d.]+)%"
                 r"（`idle_delta` ([−+]?[\\d.]+)）")
        _iitems = re.findall(_IPCT, _gip.group(2))
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)
if c.count(C_OLD) != 1:
    sys.exit(f"the idle-pair parse appears {c.count(C_OLD)} times, expected 1; "
             f"nothing written")

c = c.replace(C_OLD, C_NEW, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the idle-step list is read in percentages, as the sentence writes it.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out:
    s_ = ln.strip()
    if "idle-step sentence" in s_ or "raises the share" in s_ or "shares and deltas" in s_:
        print("  " + s_)
        i = out.index(ln)
        for nx in out[i + 1:i + 4]:
            if nx.strip().startswith(("document says", "records say")):
                print("  " + nx.strip())
print("  ...")
for ln in out[-8:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}")
