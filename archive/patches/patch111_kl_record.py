"""The KL record's numbers, and a population scoped to a record nobody can see.

`kl_length_confound.json` is cited by the document at three places and its numbers are
quoted at two. `grep` finds `286`, `kl_length` and `idle_steps_tested` **zero** times in
the checker: the `loss == beta * kl` identity, its 286 idle steps, the per-run counts
54+49+23+57+63+40, and the four `mean_kl` values that carry the whole KL/length argument
are all unread.

Worse than unread: the sentence at 5.2.4 says

    同算法同组大小的两个对照里 KL 都是下降的（GRPO 1.1068 → 1.0435，GSPO 1.1201 → 1.0217）

and 「两个」 is true **of the record** -- which holds six runs -- while there are **three**
such pairs on disk. `qwen35_9b_gspo_g4_lr1e5` (run 45) finished and is not in the record
at all, and nothing regenerates it: the tool has no caller anywhere in `tools/` or
`scripts/`. So the sentence's quantifier ranges over a set the reader cannot see, and the
reader's natural set is the run set, where the count is three. That is section 8 item 9's
exact shape -- a number byte-correct in a sentence whose scope moved -- arrived at from the
other side: not a stale numeral, but a correct numeral over an invisible population.

The guards below read the document against the record, so they are green today and fire
when the record is regenerated (which is the documented repair, and what the final pass
after the queue does). The last item is different: it fires now, because a run the
document's own run set contains is missing from the record the document cites.

Also guarded here, from the same sweep: 5.2.4's 下调 sentence, which states a `p` of 0.087,
a CI containing zero, and 「G=32 那一列」 -- a column named by group size alone, one landing
away from naming two columns. The 0.087 is `comparisons_vs_SFT.GSPO_G32_lr5e5`'s exact
McNemar p; `grep` finds it zero times in the checker too.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch111"

ANCHOR = 'g = anchor("5.2.2: the Scratch per-class maximum", P_SCRATCH)\n'

BLOCK = r'''# ------------------- the KL/length record: its runs, its 286 steps, and its KL pairs
# Cited three times, quoted twice, read never. The record's entries are keyed by the
# script's own tags (one of which, `grpo`, is not the run stem), so runs are matched to
# entries by what the entry records about itself -- group size, learning rate and IS level
# -- rather than by tag name, which the document does not pin down either.
_KL = REP / "kl_length_confound.json"
if not _KL.is_file():
    cmp("the KL/length confound record", "present", "absent",
        why="two sentences quote its numbers and a third names its field path")
else:
    _kl = json.loads(_KL.read_text(encoding="utf-8"))
    _kruns = _kl.get("runs") or {}
    _steps = {t: (e.get("kl_identity") or {}).get("idle_steps_tested")
              for t, e in _kruns.items()}
    _by_ident = {(e["G"], round(float(e["lr"]), 12), e["is_level"]): (t, e)
                 for t, e in _kruns.items()}

    def _ident(lbl):
        _m = re.fullmatch(r"([A-Za-z]+)\(G=(\d+)\) (lr[\d.eE+-]+)", lbl)
        if not _m:
            return None
        return (int(_m.group(2)), float(_m.group(3)[2:]),
                "sequence" if _m.group(1) == "GSPO" else "token")

    _unparsed = [n for n in rl if _ident(n) is None]
    cmp("every RL run's label parses into (algorithm, group size, learning rate)",
        [], _unparsed,
        why="the runs that do not would be silently outside every population computed "
            "here -- the pair sets, the KL pairs, the record's coverage")
    _covered = {n: _by_ident.get(_ident(n)) for n in rl if _ident(n)}
    _absent = [n for n in rl if _ident(n) and not _covered[n]]
    _finished = [n for n in _absent if train_record(n).is_file()]
    print(f"         the KL record covers {len(rl) - len(_absent)} of {len(rl)} RL runs")
    for _n in _absent:
        print(f"           {_n}: absent from the record"
              + (" -- FINISHED, so the record is behind the run set"
                 if _n in _finished else " (no training record yet)"))
    if _finished:
        cmp("the KL record covers every finished RL run", [], _finished,
            why="the record is one landing behind: its 六个 run and 286 空转步 describe the "
                "record, but the document's KL sentence draws a conclusion over 「同算法同组"
                "大小の对照」 -- and the run set has one more of those than the record does. "
                "The repair is to regenerate the record (tools/kl_length_confound.py has "
                "no caller; nothing regenerates it), then re-read this and the sentences "
                "below")

    # The identity and its two statements of 「六个 run 的全部 286 个空转步」.
    # `\s*` between the numeral and 个, because the document writes counts both ways:
    # 「六个 run」 with no space and 「7 个 run」 with one. A pattern that hardcodes either
    # convention matches one of them and silently reads nothing on the other -- which is
    # what cn2int's own docstring is about, and what this item's absent-branch caught.
    P_IDENT = (r"(" + NUM + r")\s*个 run 的全部 (" + NUM + r")\s*个空转步均满足（逐 run 实测为"
               r"\s*([\d+]+)")
    _gi = re.search(P_IDENT, text)
    if _gi:
        cmp("8: the KL identity's run count", cn2int(_gi.group(1)), len(_kruns),
            kind="count", pattern=P_IDENT, group=1)
        cmp("8: the KL identity's idle-step total", cn2int(_gi.group(2)),
            sum(v for v in _steps.values() if isinstance(v, int)),
            kind="count", pattern=P_IDENT, group=2)
        cmp("8: the per-run idle-step counts listed beside it",
            sorted(int(x) for x in _gi.group(3).split("+")),
            sorted(v for v in _steps.values() if isinstance(v, int)),
            why="the sentence lists one count per run; it does not claim an order, so "
                "this compares them as a bag -- a reordering of the record would "
                "otherwise fire this for a sentence that is still true")
        cmp("8: the total actually equals the per-run counts it lists",
            sum(int(x) for x in _gi.group(3).split("+")),
            sum(v for v in _steps.values() if isinstance(v, int)),
            why="a per-run list and a total in the same parenthesis, added up")
    else:
        cmp("8: the KL identity's run count and idle-step total", "the phrase", "absent",
            why="the 「初稿写 271」 correction and the identity's 286 both hang off this "
                "sentence; if it is rewritten nothing reads either")
    P_IDENT2 = (r"(" + NUM + r")\s*个 run 全部 (" + NUM + r")\s*个空转步上的逐值核对")
    _gj = re.search(P_IDENT2, text)
    if _gj:
        cmp("9: the KL identity's run count, second statement", cn2int(_gj.group(1)),
            len(_kruns), kind="count", pattern=P_IDENT2, group=1)
        cmp("9: the KL identity's idle-step total, second statement",
            cn2int(_gj.group(2)), sum(v for v in _steps.values() if isinstance(v, int)),
            kind="count", pattern=P_IDENT2, group=2)
    else:
        cmp("9: the KL identity's second run count and total", "the phrase", "absent",
            why="the same 286 is stated in two sections; only one was read before")

    # 5.2.4's KL pairs: a population over the record, not the run set.
    P_KLP = (r"同算法同组大小的(" + NUM + r")个对照里 KL 都是下降的\s*"
             r"（GRPO ([\d.]+) → ([\d.]+)，\s*GSPO ([\d.]+) → ([\d.]+)）")
    _gk = re.search(P_KLP, text)
    _cells = {}
    for _lbl in rl:
        _id = _ident(_lbl)
        if _id and _id in _by_ident:
            _cells.setdefault((_lbl.split("(")[0], _id[0]), {})[_id[1]] = _by_ident[_id][1]
    _kpairs = {k: v for k, v in _cells.items() if len(v) == 2 and
               all(isinstance(v[x].get("mean_kl"), (int, float)) for x in v)}
    if _gk:
        cmp("5.2.4: the number of same-algorithm same-group-size KL pairs",
            cn2int(_gk.group(1)), len(_kpairs),
            kind="count", pattern=P_KLP, group=1)
        _order = {"GRPO": _kpairs.get(("GRPO", 4)), "GSPO": _kpairs.get(("GSPO", 8))}
        _drop = [k for k, v in _order.items() if v is None]
        cmp("5.2.4: the two algorithms the sentence names each have such a pair", [], _drop,
            why="the sentence names GRPO and GSPO by algorithm only; if a second GSPO pair "
                "exists the name no longer identifies one pair, and the values beside it "
                "belong to whichever the writer had in mind")
        for _algo, _lo, _hi in (("GRPO", 2, 3), ("GSPO", 4, 5)):
            _cell = _order.get(_algo)
            if not _cell:
                continue
            cmp(f"5.2.4: {_algo}'s KL at the lower and higher learning rate",
                [float(_gk.group(_lo)), float(_gk.group(_hi))],
                [round(_cell[1e-05]["mean_kl"], 4), round(_cell[5e-05]["mean_kl"], 4)],
                why="the sentence's argument is that raising the learning rate lowers the "
                    "mean per-token KL; the two numbers are its evidence")
            cmp(f"5.2.4: {_algo}'s KL does fall when the learning rate rises",
                _cell[5e-05]["mean_kl"] < _cell[1e-05]["mean_kl"], True,
                why="the sentence's claim is 「都下降」, which is a comparison of the two "
                    "numbers, not their presence")
    else:
        cmp("5.2.4: the same-algorithm same-group-size KL pairs", "the phrase", "absent",
            why="four KL values and a population in one sentence; nothing reads it without "
                "this phrase")

# ---------------------------- 5.2.4: the G=32 column's evidence, quoted from the record
_PV = REP / "paired_significance.json"
if _PV.is_file():
    _pv = json.loads(_PV.read_text(encoding="utf-8"))
    _SFT = _pv.get("comparisons_vs_SFT", {})
    _g32 = [n for n in rl if "G=32" in n and "lr5e-5" in n]
    _g8 = [n for n in rl if "G=8" in n and "lr5e-5" in n]
    P_LOWER = (r"\*\*G=32 那一列的强度需要下调。\*\* 它比分低于 SFT，但成对检验 `p` =\s*"
               r"([\d.]+)、\s*置信区间\*\*包含 0\*\*")
    _gl = re.search(P_LOWER, text)
    if _gl and len(_g32) == 1 and len(_g8) == 1:
        _c = _SFT["GSPO_G32_lr5e5"]
        cmp("5.2.4: the p quoted for the G=32 column", float(_gl.group(1)),
            round(_c["mcnemar"]["p_exact_two_sided"], 3), kind="count",
            pattern=P_LOWER, group=1)
        cmp("5.2.4: that the G=32 column scores below SFT", _c["delta_vs_sft"] < 0, True,
            why="「比分低于 SFT」 is a claim about the sign of the paired delta")
        cmp("5.2.4: that its confidence interval contains zero",
            _c["delta_ci95_rows"][0] < 0 < _c["delta_ci95_rows"][1], True,
            why="computed from the interval itself rather than from the record's "
                "`excludes_zero` flag, which would be the same claim checked against "
                "itself")
        cmp("5.2.4: that this column's evidence is weaker than GSPO(G=8) lr5e-5's",
            _c["mcnemar"]["p_exact_two_sided"]
            > _SFT["GSPO_G8_lr5e5"]["mcnemar"]["p_exact_two_sided"], True,
            why="「弱于」 is a comparison of the two p values; the two runs are the only "
                "two in the sentence and both are fixed, so this is a checkable claim")
        print(f"         G=32 lr5e-5 vs SFT: p={_c['mcnemar']['p_exact_two_sided']:.6f} "
              f"ci={_c['delta_ci95_rows']} | G=8 lr5e-5: "
              f"p={_SFT['GSPO_G8_lr5e5']['mcnemar']['p_exact_two_sided']:.6f}")
    elif not _gl:
        cmp("5.2.4: the G=32 column's quoted p and interval", "the phrase", "absent",
            why="the sentence quotes a p and a CI position; both are unread without it")
    # The column is named by group size alone. Run 41 is a second G=32 run, in training.
    cmp("5.2.4: the number of columns the sentence's 「G=32」 could name",
        len([n for n in runs if "G=32" in n]), 1,
        why="「G=32 那一列」 identifies a column only while one G=32 run exists; with two, "
            "the sentence has to name the learning rate before the p beside it is usable")

'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)

for what, cond in (("the insertion point", c.count(ANCHOR) == 1),
                   ("the KL items", "kl_length" not in c and "P_IDENT" not in c)):
    if not cond:
        sys.exit(f"{what} is not as expected; nothing written")

c = c.replace(ANCHOR, BLOCK + ANCHOR, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the KL/length record is now read: its run set, its 286 idle steps, the per-run "
      "counts, the KL pairs behind the 5.2.4 argument, and the G=32 column's p.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out:
    s_ = ln.strip()
    if any(t in s_ for t in ("KL", "kl", "286", "G=32 那一列", "G=32 column", "idle-step",
                             "covers", "absent from", "parses", "p quoted")):
        print("  " + s_)
if r.stderr.strip():
    print("--- stderr ---")
    for ln in r.stderr.strip().splitlines()[-12:]:
        print("  " + ln)
print("  ...")
for ln in out[-6:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}")
if r.returncode == 1:
    print("The red items are the finding, not a bug in the patch: a finished run is "
          "missing from the KL record the document cites. The repair is to regenerate "
          "the record and re-read the sentences above it -- NOT to revert this.")
elif r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit("count-only failures (exit 2): unexpected here; RESTORED")
