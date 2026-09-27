"""The third draw's config gets the same comparison seed 2 got, against the first draw.

patch101 exists because the record used to *assert* that the second seed differed only
in the seed: a sentence, not a measurement, and one that would have read the same if
the launcher had passed a different learning rate by a copy-paste slip. The third draw
would inherit exactly that hole -- 5.2.8 has to be able to say the third run is the
same configuration as the first, and that claim has to be the result of comparing two
training records rather than a restatement of the intent behind the queue script.

The comparison is against SEED1, not against the second seed. "Same configuration" is
a statement about the cell, and the cell's owner is the first draw; comparing the
second and third draws to each other would leave open the case where both drifted the
same way. Four outcomes are recorded, as in patch101: only the seed differs; the
configs are identical including the seed (a repeat, not a reseed); something else
differs, and it is named; or a training record is missing, in which case the record
says the claim is unverified instead of leaving an assertion standing.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
TOOL = ROOT / "tools/seed_variance.py"

DESIGN_OLD = '''    print(f"  config diff (seed1 vs seed2): {_config_diff or 'none'}")'''

DESIGN_NEW = '''    print(f"  config diff (seed1 vs seed2): {_config_diff or 'none'}")

# The extra draws' configs, compared the same way and against the FIRST draw rather
# than against each other: "same configuration" is a statement about the cell, and the
# cell's owner is the first draw. Comparing the later draws to each other would leave
# open the case where both drifted the same way.
_extra_cfg = {}
for _tag, _lbl, _v, _r in EXTRA_LOADED:
    _c = config_of(_tag)
    if _c is None or not _config_compared:
        _extra_cfg[_lbl] = {
            "compared": False, "diff": None,
            "identical_except": "unverified: this draw's training record and the first "
                                "draw's are not both on disk, so the configurations "
                                "were never compared and 'the same configuration' is "
                                "an assumption, not a finding"}
        print(f"  WARNING: cannot compare {_lbl} against {SEED1[1]}; the record will "
              f"say that claim is unverified")
        continue
    _d = sorted(k for k in set(_c1) | set(_c) if _c1.get(k) != _c.get(k))
    _extra_cfg[_lbl] = {
        "compared": True,
        "diff": _d,
        "identical_except": (
            f"the random seed ({_s1} -> {_c.get('seed')}); every other config field "
            f"compared equal to {SEED1[1]} in both training records"
            if _d == ["seed"] else
            "nothing: identical to the first draw, seed included -- this is a repeat, "
            "not a reseed" if not _d else
            "the random seed, AND " + ", ".join(_d)
            + " -- not single-variable against the first draw, so what it adds to the "
              "range is not a pure seed effect"),
    }
    print(f"  config diff ({SEED1[1]} vs {_lbl}): {_d or 'none'}")'''

OUT_OLD = '''               "seeds": {"seed1": _s1, "seed2": _s2},
               "benchmark": "the same 252 classification rows for both"},'''

OUT_NEW = '''               "seeds": {"seed1": _s1, "seed2": _s2},
               "extra": _extra_cfg,
               "benchmark": "the same 252 classification rows for both"},'''

s = TOOL.read_text(encoding="utf-8")
shutil.copy2(TOOL, "/tmp/seed_variance.py.bak-patch137")
for what, old, new in (("the design block", DESIGN_OLD, DESIGN_NEW),
                       ("the record", OUT_OLD, OUT_NEW)):
    if s.count(old) != 1:
        sys.exit(f"{what}: the anchor matches {s.count(old)} times, expected 1; "
                 f"nothing written")
    s = s.replace(old, new, 1)
TOOL.write_text(s, encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/seed_variance.py.bak-patch137", TOOL)
    sys.exit(f"the patched seed_variance.py does not compile, restored:\n{r.stderr}")
print("seed_variance.py: each extra draw's config is compared against the first draw")

r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=str(ROOT))
print(f"\n--- seed_variance.py: exit {r.returncode} ---")
for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-6:]:
    print("  " + ln[:170])
