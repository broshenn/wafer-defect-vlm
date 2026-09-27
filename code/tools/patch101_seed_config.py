"""Make `seed_variance.json` compare the two configurations instead of asserting them.

The record carries this field:

    "identical_except": "the random seed; same algorithm, group size 4,
                         learning rate 1e-5, data, steps"

It is a sentence, not a measurement. Nothing in the tool looks at either run's config;
if queue 42's driver had passed `--learning_rate 5e-5` by a copy-paste slip, or a
different group size, or a different number of steps, the record would say exactly the
same thing, and 5.2.7 would tell the reader the two runs differ only in the seed. The
whole point of that run is that it changes one thing; a claim about *which* thing is the
one claim in it that has to be checked rather than declared.

This is the same failure as the reward-table lookup that ranked "lowest of N" over a
subset while the sentence said N: a number that is right, inside an artefact that cannot
show the question was ever asked.

So the comparison is computed here from the two runs' training records -- the same
records the rest of the document reads its configs from -- and the field becomes the
result of that comparison. Three outcomes, all recorded:

  * only `seed` differs            -> the claim holds, and the record says it was checked;
  * `seed` and other keys differ   -> the record names them, and says the pair is not
                                      single-variable;
  * a training record is missing   -> the record says the claim is unverified rather
                                      than leaving the assertion standing.

`seed_variance.py` has not been run yet when this is applied, so this changes what the
record will say rather than correcting a record that already exists.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/seed_variance.py"

OLD_DESIGN = '''    "design": {"seed1": SEED1[1], "seed2": SEED2[1],
               "identical_except": "the random seed; same algorithm, group size 4, "
                                   "learning rate 1e-5, data, steps",
               "benchmark": "the same 252 classification rows for both"},'''

NEW_DESIGN = '''    "design": {"seed1": SEED1[1], "seed2": SEED2[1],
               "identical_except": _identical_except,
               "config_compared": _config_compared,
               "config_diff": _config_diff,
               "seeds": {"seed1": _s1, "seed2": _s2},
               "benchmark": "the same 252 classification rows for both"},'''

ANCHOR = '''# The two facts the replication sentence is computed from.
_same_side = (acc1 - accs >= 0) == (d_vs_sft_2 >= 0)
_sig = bool(pval(ps1) < 0.05 or pval(ps) < 0.05)
'''

BLOCK = '''# The two facts the replication sentence is computed from.
_same_side = (acc1 - accs >= 0) == (d_vs_sft_2 >= 0)
_sig = bool(pval(ps1) < 0.05 or pval(ps) < 0.05)

# ------------------------------------------------------------------ the design claim
# "these two runs differ only in the seed" is the premise of everything below, so it is
# compared rather than declared. Two runs of the same configuration produce the same
# config block except for the seed; anything else that moved makes this pair a
# confounded comparison, and a confounded comparison is what the section must not call
# a noise floor.
TRAIN_RESULT = {"qwen35_9b_grpo": "grpo_train_result"}


def config_of(tag):
    """The run's config, or None if its training record is not on disk."""
    p = REPORTS / f"{TRAIN_RESULT.get(tag, tag + '_train_result')}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("config") or {}


_c1, _c2 = config_of(SEED1[0]), config_of(SEED2[0])
_s1 = (_c1 or {}).get("seed")
_s2 = (_c2 or {}).get("seed")
if _c1 is None or _c2 is None:
    _config_compared = False
    _config_diff = []
    _identical_except = ("unverified: the two training records are not both on disk, so "
                         "the configurations were never compared and 'only the seed "
                         "differs' is an assumption, not a finding")
    print("  WARNING: cannot compare the two configs (a training record is missing); "
          "the record will say the design claim is unverified")
else:
    _config_compared = True
    _config_diff = sorted(k for k in set(_c1) | set(_c2) if _c1.get(k) != _c2.get(k))
    if _config_diff == ["seed"]:
        _identical_except = (f"the random seed ({_s1} -> {_s2}); every other config "
                             f"field compared equal in both training records")
    elif not _config_diff:
        _identical_except = ("nothing: the two runs' configs are identical, seed "
                             "included -- this is a repeat, not a reseed")
    else:
        _identical_except = ("the random seed, AND " + ", ".join(_config_diff)
                             + " -- the pair is not single-variable, so the difference "
                               "below is not a pure seed effect")
    print(f"  config diff (seed1 vs seed2): {_config_diff or 'none'}")
'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/seed_variance.py.bak-patch101")
for old, new, what in ((OLD_DESIGN, NEW_DESIGN, "the design block"),
                       (ANCHOR, BLOCK, "the insertion point")):
    if s.count(old) != 1:
        sys.exit(f"{what} appears {s.count(old)} times, expected 1; nothing written")
    s = s.replace(old, new, 1)
TOOL.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/seed_variance.py.bak-patch101", TOOL)
    sys.exit(f"the patched seed_variance.py does not compile, restored:\\n{r.stderr}")
print("seed_variance.py now compares the two configs and records the result")
