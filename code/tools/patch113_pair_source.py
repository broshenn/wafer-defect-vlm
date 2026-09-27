"""The pairs come from two places in the record, and they have to agree.

patch112 crashed with `KeyError: 'completion_length'`, and the crash is worth more than
the fix. The record holds the pairs twice:

  * `runs.<tag>` -- one entry per run, carrying `mean_kl` and the `kl_identity` block.
    This is what the checker's own pair population is computed from: two runs with the
    same (algorithm, group size) and different learning rates.
  * `matched_pairs.<"GRPO G=4">` -- one entry per pair, carrying the *comparison*:
    `kl` and its delta, `completion_length` and its percentage, `idle` and its delta.

The KL values are in both; the lengths and the idle shares are only in the second. So the
population that the sentences talk about (「同算法同组大小的三个对照」) is assembled by the
checker from one half of the record and quoted by the document from the other half, and
nothing checked that the two halves name the same pairs. If `matched_pairs` were missing a
pair that `runs` implies, every length and idle item would silently read a smaller set --
which is the shape this whole family of defects has: a correct number over a population
that is not the one the sentence names.

So: build the pair map from `matched_pairs`, assert it covers the pairs the runs imply, and
read the lengths and the idle shares from it. `mean_kl` stays read from `runs`, so the KL
items and the length items do not come from the same place.
"""
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch113"

C_OLD = '''    # What the record says about each pair, in the shapes the sentences quote it in.
    _rec_kl = {k: (round(v[1e-05]["mean_kl"], 4), round(v[5e-05]["mean_kl"], 4))
               for k, v in _kpairs.items()}
    _rec_len = {k: (round(v[1e-05]["completion_length"], 1),
                    round(v[5e-05]["completion_length"], 1))
                for k, v in _kpairs.items()}
    # Section 8 quotes the lengths exactly, so the percentage printed beside them is
    # required to follow from them; 5.2.4 quotes them to 1 dp and prints no percentage.
    _rec_len2 = {k: (round(v[1e-05]["completion_length"], 2),
                     round(v[5e-05]["completion_length"], 2),
                     round(100 * (v[5e-05]["completion_length"]
                                  / v[1e-05]["completion_length"] - 1), 1))
                 for k, v in _kpairs.items()}
    _rec_idle = {k: (round(100 * v[1e-05]["idle"], 2), round(100 * v[5e-05]["idle"], 2),
                     round(v[5e-05]["idle"] - v[1e-05]["idle"], 4))
                 for k, v in _kpairs.items()}
'''

C_NEW = '''    # The record's own pair list. `_kpairs` above is assembled from `runs.<tag>` entries
    # (which carry `mean_kl`); the lengths and the idle shares exist only per pair, under
    # `matched_pairs`, whose keys are the script's names for them ("GRPO G=4"). Reading a
    # length out of a run entry is a KeyError -- which is how this item came to exist.
    _mp = {}
    for _nm, _e in (_kl.get("matched_pairs") or {}).items():
        _mm = re.fullmatch(r"([A-Za-z]+) G=(\\d+)", _nm)
        if _mm:
            _mp[(_mm.group(1), int(_mm.group(2)))] = _e
    cmp("the record's pair list covers every pair its runs imply",
        [], sorted(set(_kpairs) - set(_mp)),
        why="the document's population sentence counts pairs; the count is assembled from "
            "the runs while the lengths and idle shares it quotes come from the pair "
            "entries. If the two lists disagree, every item below reads a smaller "
            "population than the sentence names and reports green")
    _mp2 = {k: v for k, v in _mp.items() if k in _kpairs}
    # What the record says about each pair, in the shapes the sentences quote it in.
    _rec_kl = {k: (round(v[1e-05]["mean_kl"], 4), round(v[5e-05]["mean_kl"], 4))
               for k, v in _kpairs.items()}
    _rec_len = {k: (round(v[1e-05]["completion_length"], 1),
                    round(v[5e-05]["completion_length"], 1))
                for k, v in _mp2.items()}
    # Section 8 quotes the lengths exactly, so the percentage printed beside them is
    # required to follow from them; 5.2.4 quotes them to 1 dp and prints no percentage.
    _rec_len2 = {k: (round(v[1e-05]["completion_length"], 2),
                     round(v[5e-05]["completion_length"], 2),
                     round(100 * (v[5e-05]["completion_length"]
                                  / v[1e-05]["completion_length"] - 1), 1))
                 for k, v in _mp2.items()}
    _rec_idle = {k: (round(100 * v[1e-05]["idle"], 2), round(100 * v[5e-05]["idle"], 2),
                     round(v[5e-05]["idle"] - v[1e-05]["idle"], 4))
                 for k, v in _mp2.items()}
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)
if c.count(C_OLD) != 1:
    sys.exit(f"the pair-source block appears {c.count(C_OLD)} times, expected 1; "
             f"nothing written")

c = c.replace(C_OLD, C_NEW, 1)
# `re` and `_kl` are both live in this scope: `re` is a module import at the top and `_kl`
# is the parsed record, which the `if _KL.is_file():` branch above guarantees.
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the pair lists are reconciled: the lengths and idle shares now read from the "
      "record's own pair entries, with an item for the two lists disagreeing.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out:
    s_ = ln.strip()
    if ("pair" in s_ or "KL" in s_ or "completion" in s_ or "enumerates" in s_
            or "raises the share" in s_ or "idle-step" in s_):
        print("  " + s_)
if r.stderr.strip():
    print("--- stderr ---")
    for ln in r.stderr.strip().splitlines()[-12:]:
        print("  " + ln)
print("  ...")
for ln in out[-6:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}")
