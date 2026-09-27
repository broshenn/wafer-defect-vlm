"""Make kl_length_confound.py extend itself instead of narrowing silently.

Two hand-written snapshots live in this tool:

  RUNS   -- six training records. LIMITATIONS quotes this record for a claim about
            *all* runs ("the identity holds on every idle step of the six runs,
            54+49+23+57+63+40"). A seventh run that lands without being added here
            does not make the record wrong; it makes it narrower than the sentences
            that cite it, which is the defect section 8 item 9 keeps recording.
  pairs  -- two matched lr pairs. The tool's own conclusion says "in both pairs",
            a quantifier that goes false the moment a third pair exists.

Both now extend from what is on disk:

  * a *_train_result.json whose outcome is `completed` and whose steps_logged equals
    its max_steps_requested, and that is not `_train_result` of SFT, is added to
    RUNS. A run still in flight is excluded -- it is not one of "all runs" yet.
    Each auto-added tag is printed, so the record cannot grow without saying so.
  * a matched lr pair is derived for every (algorithm, group size) cell whose two
    learning rates both have a record. Only the fully-qualified tag form
    (qwen35_9b_gspo_g4_lr1e5) is parsed for this; the three legacy tags that omit
    their group size or lr are left to the explicit list, because guessing a
    default for them is how a record ends up describing a run that never ran.
    The seed run is excluded by the same rule: it repeats a cell, it is not a
    second lr.
  * the conclusion's quantifier is counted from the pairs it actually has.

Today this changes no numbers: six runs and two pairs are already complete, and the
rehearsal is that the `runs` and `matched_pairs` blocks come out byte-identical.
"""
import ast
import json
import re
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/kl_length_confound.py")
s = P.read_text(encoding="utf-8")

# ------------------------------------------------------- 1. the run set extends
A1 = '''RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",
        "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g4_lr5e5"]
'''
A1_NEW = '''RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",
        "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g4_lr5e5"]

REPORTS = ROOT / "outputs/reports"
SFT_RECORD = "sft_train_result.json"

# A finished run on disk that RUNS does not name is added, not skipped. "Finished"
# is the record's own claim (outcome completed, steps_logged == max_steps_requested):
# a run still training has a log and a partial record, and counting it as one of
# "all runs" would make every sentence that says "all runs" wrong in a new way.
for _p in sorted(REPORTS.glob("*_train_result.json")):
    _tag = _p.name[: -len("_train_result.json")]
    if _p.name == SFT_RECORD or _tag in RUNS:
        continue
    try:
        _rec = json.loads(_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"NOTE: {_p.name} is unreadable; not included", file=sys.stderr)
        continue
    if (_rec.get("outcome") == "completed"
            and _rec.get("steps_logged") == _rec.get("max_steps_requested")):
        RUNS.append(_tag)
        print(f"NOTE: {_tag} finished and was not in RUNS; included. "
              f"Add it to RUNS so the record's order is a decision, not a sort.",
              file=sys.stderr)
    else:
        _shape = f"outcome={_rec.get('outcome')} steps={_rec.get('steps_logged')}/" \\
                 f"{_rec.get('max_steps_requested')}"
        print(f"NOTE: {_tag} is on disk but not finished ({_shape}); excluded from "
              f"a record whose claims say 'all runs'", file=sys.stderr)
'''

# ------------------------------------------------- 2. the pairs derive, and so
#                                                     does the quantifier
A2 = '''pairs = [("GRPO G=4", "grpo", "qwen35_9b_grpo_lr5e5"),
         ("GSPO G=8", "gspo_lr1e5", "qwen35_9b_gspo_v1")]
'''
A2_NEW = '''pairs = [("GRPO G=4", "grpo", "qwen35_9b_grpo_lr5e5"),
         ("GSPO G=8", "gspo_lr1e5", "qwen35_9b_gspo_v1")]

# Every (algorithm, group size) cell that has both learning rates becomes a pair.
CELL = re.compile(r"^qwen35_9b_(gspo|grpo)_g(\\d+)_lr([0-9.eE+-]+)$")
_cells = {}
for _tag in RUNS:
    _m = CELL.match(_tag)
    if not _m:
        continue        # a legacy tag or a seed repeat; the explicit list owns those
    _algo, _g, _lr = _m.group(1).upper(), int(_m.group(2)), float(_m.group(3))
    _cells.setdefault((_algo, _g), {})[_lr] = _tag
for (_algo, _g), _bys in sorted(_cells.items()):
    if len(_bys) != 2:
        continue
    _lo, _hi = sorted(_bys)
    _label = f"{_algo} G={_g}"
    if _label in [n for n, _, _ in pairs]:
        continue
    pairs.append((_label, _bys[_lo], _bys[_hi]))
'''
if s.count(A2) != 1:
    sys.exit(f"A2 anchor appears {s.count(A2)} times")

# ---------------------------------------------------- 3. count, do not assert
A3 = '''out["conclusion"] = (
    "kl is a per-token mean -- the identity loss == beta*kl holds on every idle "
    "step of every run, to within the logs' own rounding -- so it is not scaled "
    "by completion length. But the completions are not the same length across "
    "the pairs being compared: the 5e-5 run emits measurably shorter completions "
    "in both pairs, so its mean is taken over a different span and the two "
    "values do not measure the same quantity. The raw kl falls in both pairs, "
    "which is the opposite of what LIMITATIONS 5.2.2 claims; but this "
    "measurement cannot establish the direction either way, so the claim is "
    "removed rather than reversed. What IS consistent across both pairs at "
    "5e-5: reward falls and completions shorten.")'''
A3_NEW = '''# "in both pairs" was a quantifier written when there were two. It is counted now,
# and the two per-pair facts behind it are read off the pairs rather than asserted.
_short = [n for n, d in out["matched_pairs"].items() if d["length_delta"] < 0]
_falls = [n for n, d in out["matched_pairs"].items() if d["reward_delta"] < 0]
_n = len(out["matched_pairs"])
out["conclusion"] = (
    "kl is a per-token mean -- the identity loss == beta*kl holds on every idle "
    "step of every run, to within the logs' own rounding -- so it is not scaled "
    "by completion length. But the completions are not the same length across "
    "the pairs being compared: of the " + str(_n) + " matched lr pairs, the 5e-5 "
    "run emits shorter completions in " + str(len(_short)) + " ("
    + (", ".join(_short) if _short else "none") + "), so its mean is taken over a "
    "different span and the two values do not measure the same quantity. The raw "
    "kl falls in " + str(len(_short)) + " of them, which is the opposite of what "
    "LIMITATIONS 5.2.2 claims; but this measurement cannot establish the direction "
    "either way, so the claim is removed rather than reversed. What the pairs "
    "agree on at 5e-5: the mean reward is lower in " + str(len(_falls)) + " of "
    + str(_n) + " (" + (", ".join(_falls) if _falls else "none") + ").")'''
if s.count(A3) != 1:
    sys.exit(f"A3 anchor appears {s.count(A3)} times")

for old, new in ((A1, A1_NEW), (A2, A2_NEW), (A3, A3_NEW)):
    s = s.replace(old, new, 1)
s = s.replace("import statistics\nimport sys", "import statistics\nimport sys", 1)

try:
    ast.parse(s)
except SyntaxError as e:
    sys.exit(f"edited file does not parse: {e}; nothing written")
for want in ("REPORTS = ROOT", "CELL = re.compile", "_falls = ["):
    if want not in s:
        sys.exit(f"edited file is missing {want!r}; nothing written")

P.write_text(s, encoding="utf-8")
print(f"kl_length_confound.py: RUNS and pairs now extend from disk "
      f"({len(s.splitlines())} lines, parses)")
