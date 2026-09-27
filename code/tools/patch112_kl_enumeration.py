"""Read the pair enumerations by parsing them, not by counting them.

Two things found by regenerating the KL record, both the same defect from the two sides
the document and its reader can be wrong from.

1. The KL sentence at 5.2.4 now enumerates three pairs and names each one
   `ALGO(G=n) lo → hi`. The item that read it keyed on positional groups
   (`GRPO lo hi，GSPO lo hi`) and so reported **「absent」** the moment the sentence named a
   third pair -- the absent-branch doing exactly what it exists for. A pattern that stops
   matching and reports a fact is a broken probe, not a stale document, and the repair is
   to read the enumeration rather than to trust group numbers. The rewrite below parses
   every `ALGO(G=n) lo → hi` inside the parenthesis into a dict keyed by (algorithm, group
   size), which is what patch109 already does for the same population in 5.2.2, and which
   makes the item independent of how many pairs there are and of the order they are named
   in. The enumerated **set** is compared as a fact, so a landing that grows the population
   turns the fact red and `--fix` will not bump the numeral while it is.

2. Section 8 (line 903) states the same population a fourth time:

       两个**同算法同组大小**的对照方向相反：
       GRPO(G=4) 36.00% → 32.67%（`idle_delta` −0.0333），
       GSPO(G=8) 15.33% → 38.00%（`idle_delta` +0.2267）。
       即「学习率把空转步推高」在 G=8 上成立、在 G=4 上不成立。

   Nobody reads it: patch109's two patterns both require 条 and a 、 after 同算法, and this
   one says 个 and runs 同算法同组大小 together. And its 「两个」 is now wrong the same way
   5.2.4's 两个 was: three such pairs exist, and the third (GSPO(G=4) 33.33% → 26.67%,
   −0.0667) lowers the share, so it belongs to the sentence's own conclusion -- 「在 G=4 上
   不成立」 now has two instances rather than one. The sentence is rewritten to name all
   three, and read here: the count, the three (rate, rate, delta) triples, and the
   conclusion as a set comparison.

Also read: the completion lengths beside the KL pairs in 5.2.4 and again in section 8 with
their percentage changes. The percentages are **computed** from the two lengths rather than
read from `length_change_pct`, so the sentence is checked against the arithmetic a reader
would do, not against the record's own summary of it.
"""
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
DOC = ROOT / "LIMITATIONS.md"
BAK_C = "/tmp/check_quantified_claims.py.bak-patch112"
BAK_D = "/tmp/LIMITATIONS.md.bak-patch112"

# ---------------------------------------------------------------- the document, section 8
DOC_OLD = """   两个**同算法同组大小**的对照方向相反：
   GRPO(G=4) 36.00% → 32.67%（`idle_delta` −0.0333），
   GSPO(G=8) 15.33% → 38.00%（`idle_delta` +0.2267）。
   即「学习率把空转步推高」在 G=8 上成立、在 G=4 上不成立。"""
DOC_NEW = """   三个**同算法同组大小**的对照方向并不一致：
   GSPO(G=8) 15.33% → 38.00%（`idle_delta` +0.2267）、
   GRPO(G=4) 36.00% → 32.67%（`idle_delta` −0.0333）、
   GSPO(G=4) 33.33% → 26.67%（`idle_delta` −0.0667）。
   即「学习率把空转步推高」只在 G=8 上成立，两个 G=4 的对照都把它压低。"""

# ----------------------------------------- the document, section 8, the length precision
# `109.2 / 114.4 - 1 = -4.5%` -- but the sentence prints -4.6%, which is the record's
# `length_change_pct` computed from the unrounded 109.16. The printed pair does not
# reproduce the printed percentage, which is section 8 item 9's shape in miniature: a
# number that is right about the record and not about the sentence beside it. Of the three
# pairs this is the only one where the last digit of the rounding changes the percentage.
# The fix is precision rather than a different number: section 8 is the audit and already
# quotes exact values elsewhere, so it quotes the lengths exactly here and the item below
# then requires every printed percentage to follow from the pair printed beside it.
DOC_OLD_2 = """   （GRPO(G=4) 115.5 → 97.9 token，−15.2%；GSPO(G=8) 114.4 → 109.2，−4.6%；
   GSPO(G=4) 117.0 → 113.8，−2.7%），"""
DOC_NEW_2 = """   （GRPO(G=4) 115.48 → 97.92 token，−15.2%；GSPO(G=8) 114.40 → 109.16，−4.6%；
   GSPO(G=4) 116.97 → 113.83，−2.7%；此处取两位小数是必需的——1 位小数下
   GSPO(G=8) 那一对的 −4.6% 无法从相邻的两个数算出来），"""

# ------------------------------------------------------------------- the checker, region 1
C_OLD_1 = '''    P_KLP = (r"同算法同组大小的(" + NUM + r")个对照里 KL 都是下降的\\s*"
             r"（GRPO ([\\d.]+) → ([\\d.]+)，\\s*GSPO ([\\d.]+) → ([\\d.]+)）")
    _gk = re.search(P_KLP, text)
    _cells = {}
    for _lbl in rl:
        _id = _ident(_lbl)
        if _id and _id in _by_ident:
            _cells.setdefault((_lbl.split("(")[0], _id[0]), {})[_id[1]] = _by_ident[_id][1]
    _kpairs = {k: v for k, v in _cells.items() if len(v) == 2 and
               all(isinstance(v[x].get("mean_kl"), (int, float)) for x in v)}
'''

C_NEW_1 = '''    # `ALGO(G=n) lo → hi`, repeated. Read by parsing every occurrence inside the
    # parenthesis into a dict keyed by (algorithm, group size) -- the same way patch109
    # reads this population in 5.2.2. Positional groups were the first attempt and they
    # reported the sentence "absent" as soon as it named a third pair: a pattern that
    # stops matching and reports a fact is a broken probe, not a stale document.
    _PAIR_ITEM = r"([A-Za-z]+)\\(G=(\\d+)\\) ([\\d.]+) → ([\\d.]+)"

    def _parsed(s):
        return {(m.group(1), int(m.group(2))): (float(m.group(3)), float(m.group(4)))
                for m in re.finditer(_PAIR_ITEM, s)}

    _cells = {}
    for _lbl in rl:
        _id = _ident(_lbl)
        if _id and _id in _by_ident:
            _cells.setdefault((_lbl.split("(")[0], _id[0]), {})[_id[1]] = _by_ident[_id][1]
    _kpairs = {k: v for k, v in _cells.items() if len(v) == 2 and
               all(isinstance(v[x].get("mean_kl"), (int, float)) for x in v)}
    # What the record says about each pair, in the shapes the sentences quote it in.
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
    P_KLP = r"同算法同组大小的(" + NUM + r")个对照里 KL 都是下降的\\s*（([^）]*)）"
    _gk = re.search(P_KLP, text)
'''

# ------------------------------------------------------------------- the checker, region 2
C_OLD_2 = '''    if _gk:
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
'''

C_NEW_2 = '''    if _gk:
        _named = _parsed(_gk.group(2))
        cmp("5.2.4: the number of same-algorithm same-group-size KL pairs",
            cn2int(_gk.group(1)), len(_kpairs),
            kind="count", pattern=P_KLP, group=1)
        cmp("5.2.4: which pairs the KL sentence enumerates", sorted(_named),
            sorted(_rec_kl),
            why="the numeral and the list are one claim. A landing that grows the "
                "population must name the new pair, and until it is named this fact is "
                "red -- which is what stops `--fix` from bumping the numeral alone, and "
                "is the intended outcome: which pairs the sentence is evidence about is "
                "a decision, not a count")
        for _k in sorted(_named):
            cmp(f"5.2.4: {_k[0]}(G={_k[1]})'s KL at the lower and higher learning rate",
                list(_named[_k]), list(_rec_kl.get(_k, ())),
                why="the sentence's argument is that raising the learning rate lowers the "
                    "mean per-token KL; these two numbers are its evidence")
            cmp(f"5.2.4: {_k[0]}(G={_k[1]})'s KL does fall when the learning rate rises",
                _rec_kl.get(_k, (0.0, 0.0))[1] < _rec_kl.get(_k, (0.0, 0.0))[0], True,
                why="「都下降」 compares the two numbers; it is stated of every pair the "
                    "sentence names, so a pair enters the claim when it is written down")
    else:
        cmp("5.2.4: the same-algorithm same-group-size KL pairs", "the phrase", "absent",
            why="the sentence states a population and then names its members; without this "
                "phrase nothing reads either")

    # The other measurement of the same pairs: the completions the sentence calls shorter.
    P_KLP_LEN = r"5e-5 的补全却明显更短\\s*（([^）]*)）"
    _gkl = re.search(P_KLP_LEN, text)
    if _gkl:
        cmp("5.2.4: the completion lengths enumerated beside the KL pairs",
            _parsed(_gkl.group(1)), _rec_len,
            why="the sentence's point is that the KL and the length are not the same "
                "quantity; that needs both measurements read, not just the KL")
    else:
        cmp("5.2.4: the completion lengths beside the KL pairs", "the phrase", "absent",
            why="the KL sentence's second half; if it is rewritten nothing reads the "
                "lengths it quotes")

    # Section 8 restates the lengths, and adds the percentage change. The percentage is
    # computed from the two rounded lengths -- the arithmetic a reader would do -- rather
    # than read from the record's `length_change_pct`, which would be the record's own
    # summary of the same pair of numbers, i.e. the claim checked against itself.
    P_LEN_PCT = r"但 5e-5 那(" + NUM + r")次的补全明显更短\\s*（([^）]*)）"
    _gl = re.search(P_LEN_PCT, text)
    if _gl:
        _litems = re.findall(_PAIR_ITEM + r"(?: token)?，\\s*(−?[\\d.]+)%", _gl.group(2))
        _ltxt = {(m[0], int(m[1])): (float(m[2]), float(m[3]), -float(m[4].lstrip("−")))
                 for m in _litems}
        _lwant = _rec_len2
        cmp("8: the number of pairs whose completions the sentence calls shorter",
            cn2int(_gl.group(1)), len(_ltxt), kind="count", pattern=P_LEN_PCT, group=1)
        cmp("8: the completion lengths and the percentage changes beside them", _ltxt,
            _lwant,
            why="the percentage is computed here from the two lengths rather than read "
                "from the record's `length_change_pct`, so the item is the arithmetic a "
                "reader would do with the printed pair -- and comparing the lengths "
                "themselves is what makes that arithmetic reproducible: 1 dp is not "
                "enough precision for the GSPO(G=8) pair, whose -4.6% follows from "
                "109.16/114.40 but not from 109.2/114.4")
    else:
        cmp("8: the completion lengths with their percentage changes", "the phrase",
            "absent",
            why="this is where the lengths are quoted with a percentage; the arithmetic "
                "is only checkable while the phrase is found")

    # Section 8 states this population once more, in words patch109's two patterns do not
    # reach (they require 条 and a 、 after 同算法; this sentence says 个 and runs the two
    # words together). Its 两个 was stale the same way 5.2.4's was, and the third pair
    # belongs to its conclusion.
    P_IDLE_PAIRS = (r"(" + NUM + r")个\\*\\*同算法同组大小\\*\\*的对照方向[^：]*：\\s*"
                    r"([^。]*)。")
    _gip = re.search(P_IDLE_PAIRS, text)
    if _gip:
        _iitems = re.findall(_PAIR_ITEM + r"%（`idle_delta` ([−+]?[\\d.]+)）", _gip.group(2))
        _itxt = {(m[0], int(m[1])): (float(m[2]), float(m[3]),
                                    float(m[4].lstrip("−").replace("+", ""))
                                    * (-1 if m[4].startswith("−") else 1))
                 for m in _iitems}
        cmp("8: the number of same-algorithm same-group-size idle-step pairs",
            cn2int(_gip.group(1)), len(_kpairs), kind="count",
            pattern=P_IDLE_PAIRS, group=1)
        cmp("8: which pairs the idle-step sentence enumerates", sorted(_itxt),
            sorted(_rec_idle),
            why="same population, third statement, first reader")
        cmp("8: the idle-step shares and deltas it quotes for them", _itxt, _rec_idle,
            why="the passage's point is that the direction is not the learning rate's "
                "alone, which is a claim about these deltas")
        cmp("8: the pairs its conclusion says the learning rate raises the share in",
            sorted([k for k, v in _rec_idle.items() if v[2] > 0]),
            sorted([k for k, v in _itxt.items() if v[2] > 0]),
            why="「只在 G=8 上成立」 is a set, and it is the sentence's conclusion; the "
                "third pair lowers the share, so it belongs to this set too")
    else:
        cmp("8: the same-algorithm same-group-size idle-step pairs", "the phrase", "absent",
            why="a fourth statement of the pair population, read by none of the others")

'''

doc = DOC.read_text(encoding="utf-8")
c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(DOC, BAK_D)
shutil.copy2(CHECKER, BAK_C)

for what, cond in (("the section-8 sentence", doc.count(DOC_OLD) == 1),
                   ("the section-8 length list", doc.count(DOC_OLD_2) == 1),
                   ("the checker's P_KLP definition", c.count(C_OLD_1) == 1),
                   ("the checker's P_KLP items", c.count(C_OLD_2) == 1)):
    if not cond:
        sys.exit(f"{what} is not as expected; nothing written")

doc = doc.replace(DOC_OLD, DOC_NEW, 1)
doc = doc.replace(DOC_OLD_2, DOC_NEW_2, 1)
c = c.replace(C_OLD_1, C_NEW_1, 1)
c = c.replace(C_OLD_2, C_NEW_2, 1)
CHECKER.write_text(c, encoding="utf-8")
DOC.write_text(doc, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK_C, CHECKER)
    shutil.copy2(BAK_D, DOC)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the pair enumerations are now read by parsing them, in three places, and section 8 "
      "names all three pairs.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out:
    s_ = ln.strip()
    if ("KL" in s_ or "completion" in s_ or "idle-step pairs" in s_
            or "enumerates" in s_ or "percentage" in s_ or "raises the share" in s_):
        print("  " + s_)
if r.stderr.strip():
    print("--- stderr ---")
    for ln in r.stderr.strip().splitlines()[-12:]:
        print("  " + ln)
print("  ...")
for ln in out[-6:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}")
if r.returncode != 0:
    print("RED. Reading the tail above: if a fact about the run set moved, the document "
          "is describing the run set it had; if one of the new items disagrees with the "
          "record, the rewrite and the record disagree and that is the finding.")
