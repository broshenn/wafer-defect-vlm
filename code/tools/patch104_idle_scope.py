"""A quantifier in 5.2.5 moved when run 45 landed, and the checker could not see it.

The sentence reads:

    **空转步与基准分的读数再次相反**：G=32 的空转步 42.00%（63/150）是六组里最差的，
    但其 macro-F1 又高于空转步只有 38.00% 的 G=8 lr5e-5。

42.00% is still the largest idle-step rate of any run, so the *claim* survives -- but
"六组" was written when six runs had idle-step data. Run 45 (GSPO(G=4) lr1e-5) landed
after it, the idle-step table grew a seventh column (33.33%), and this sentence kept
saying six. The number is right, the comparison is right, and the quantifier over which
the comparison is made no longer matches the table beside it. That is section 8 item 9
verbatim, in a sentence the claims checker does not read: nothing in the tool anchors on
the idle-step population, so a green run certified it.

Two things, in this order:

1. The numeral is corrected from the records. `mean_frac_reward_zero_std` is a top-level
   field of each training record, recomputed from that run's own log rather than copied
   from prose (`tools/idle_step_table.py` checks the log and the record against each
   other three ways), so the population is "the runs whose record carries that field".
   The patch refuses if the sentence's own rate is not that population's maximum, or if
   the maximum belongs to a run other than the one the sentence names -- those are
   rewrites, not numeral changes.

2. The checker gains the item that reads it, so the next landing moves this sentence
   loudly instead of silently. The population and the holder are both checked; the
   population is a count (a new run's column makes the sentence wrong by a word) and the
   holder is a fact.

The fix is applied before the check, so the checker has to come back green on the
document this patch just wrote -- if it does not, the patch restores the file.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
CHECKER = ROOT / "tools/check_quantified_claims.py"
REP = ROOT / "outputs/reports"

CN = "一二三四五六七八九十"
NUM = r"(?:[一二三四五六七八九十两]+|\d+)"
WRONG_RUN = "GSPO(G=32) lr5e-5"

# Display name -> training record, matching the checker's own TRAIN_RESULT map.
TRAIN_RESULT = {
    "GRPO(G=4) lr1e-5": "grpo_train_result",
    "GRPO(G=4) lr5e-5": "qwen35_9b_grpo_lr5e5_train_result",
    "GSPO(G=8) lr5e-5": "qwen35_9b_gspo_v1_train_result",
    "GSPO(G=8) lr1e-5": "gspo_lr1e5_train_result",
    "GSPO(G=32) lr5e-5": "qwen35_9b_gspo_g32_train_result",
    "GSPO(G=4) lr5e-5": "qwen35_9b_gspo_g4_lr5e5_train_result",
    "GSPO(G=4) lr1e-5": "qwen35_9b_gspo_g4_lr1e5_train_result",
    "GSPO(G=32) lr1e-5": "qwen35_9b_gspo_g32_lr1e5_train_result",
}


def idle_rates():
    """The idle-step population, from the training records that carry the field."""
    out = {}
    for name, stem in TRAIN_RESULT.items():
        p = REP / f"{stem}.json"
        if not p.is_file():
            continue
        v = json.loads(p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
        if isinstance(v, (int, float)):
            out[name] = v
    return out


def cn2int(s):
    if s.isdigit():
        return int(s)
    d = {c: i + 1 for i, c in enumerate(CN)}
    d["两"] = 2
    if s == "十":
        return 10
    if "十" in s:
        a, _, b = s.partition("十")
        return (d.get(a, 1) if a else 1) * 10 + (d[b] if b else 0)
    return d[s]


rates = idle_rates()
if not rates:
    sys.exit("no training record carries mean_frac_reward_zero_std; nothing written")
worst = max(rates, key=rates.get)
n = len(rates)
if n > len(CN):
    sys.exit(f"{n} runs need a numeral this patch cannot spell; nothing written")

# ------------------------------------------------------------------ 1. the document
P_IDLE = re.compile(r"G=32 的空转步 ([\d.]+)%（(\d+)/(\d+)）是(" + NUM + r")组里最差的")
s = DOC.read_text(encoding="utf-8")
m = P_IDLE.search(s)
if not m:
    sys.exit("the idle-step sentence is not in the form this patch reads; nothing written")
if worst != WRONG_RUN:
    sys.exit(f"the worst idle-step rate now belongs to {worst}, not {WRONG_RUN}; the "
             f"sentence names a run, so this is a rewrite and not a numeral change -- "
             f"nothing written")
if abs(float(m.group(1)) - round(100 * rates[WRONG_RUN], 2)) > 1e-9:
    sys.exit(f"the sentence says {m.group(1)}% and the record says "
             f"{100 * rates[WRONG_RUN]:.4f}%; nothing written")
if cn2int(m.group(4)) == n:
    print(f"the idle-step population already reads {n} (the record's count); "
          f"nothing written")
    sys.exit(0)

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch104")
before_bold = s.count("**")
s = s[:m.start(4)] + CN[n - 1] + s[m.end(4):]
after_bold = s.count("**")
if (after_bold - before_bold) % 2:
    sys.exit(f"bold markers went {before_bold} -> {after_bold}; nothing written")
DOC.write_text(s, encoding="utf-8")
print(f"5.2.5: the idle-step population {m.group(4)} -> {CN[n - 1]} "
      f"(the {n} runs whose record carries the rate; worst is {worst} at "
      f"{100 * rates[worst]:.2f}%)")

# -------------------------------------------------------------------- 2. the checker
OLD_TAIL = '''    cmp("5.2.5: the number of columns 'lowest' is measured over",
        cn2int(g.group(2)), len(r_cols), kind="count",
        pattern=P_REWARD, group=2)

# ------------------------------------------------------------------------ verdict'''

NEW_TAIL = '''    cmp("5.2.5: the number of columns 'lowest' is measured over",
        cn2int(g.group(2)), len(r_cols), kind="count",
        pattern=P_REWARD, group=2)

# ------------------- 5.2.5: the idle-step rate, its worst run, and its population
# Nothing read this sentence, so when run 45 added a seventh idle-step column the
# "六组" beside it stayed six and the tool said ok. The population is not the run set in
# general: it is the runs whose training record carries the rate, which is the same set
# the 5.2.2 idle-step row has a column for.
P_IDLE = (r"G=32 的空转步 ([\\d.]+)%（(\\d+)/(\\d+)）是(" + NUM + r")组里最差的")
g = anchor("5.2.5: the idle-step population and its worst run", P_IDLE)
if g:
    idle = {}
    for _n in rl:
        _p = train_record(_n)
        if not _p.is_file():
            continue
        _v = json.loads(_p.read_text(encoding="utf-8")).get("mean_frac_reward_zero_std")
        if isinstance(_v, (int, float)):
            idle[_n] = _v
    if idle:
        _worst = max(idle, key=idle.get)
        cmp("5.2.5: the run with the worst idle-step rate", _worst, "GSPO(G=32) lr5e-5",
            why="the sentence names a run; a different run holding the maximum makes it "
                "false in a way a numeral cannot fix")
        cmp("5.2.5: the worst idle-step rate", float(g.group(1)),
            round(100 * idle[_worst], 2))
        cmp("5.2.5: the idle-step population", cn2int(g.group(4)), len(idle),
            kind="count", pattern=P_IDLE, group=4)
        print(f"         idle-step rates: "
              + "、".join(f"{_k} {100 * _v:.2f}%" for _k, _v in
                          sorted(idle.items(), key=lambda kv: -kv[1])))

# ------------------------------------------------------------------------ verdict'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, "/tmp/check_quantified_claims.py.bak-patch104")
if c.count(OLD_TAIL) != 1:
    sys.exit(f"the checker's reward block appears {c.count(OLD_TAIL)} times, expected 1; "
             f"the document was already written, the checker was not changed")
if "P_IDLE" in c:
    sys.exit("the checker already has an idle-step item; nothing written")
c = c.replace(OLD_TAIL, NEW_TAIL, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch104", CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

# The fix and its guard have to agree: the checker must be green on what was just
# written. It was red before this patch, on exactly this numeral, which is the point.
print("\n--- re-running the checker on the corrected document ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
for ln in r.stdout.strip().splitlines()[-12:]:
    print("  " + ln)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch104", CHECKER)
    shutil.copy2("/tmp/LIMITATIONS.md.bak-patch104", DOC)
    sys.exit(f"\nthe checker is not green after the fix (exit {r.returncode}); BOTH files "
             f"restored")
print("\nchecker green: the idle-step population is now read from the records, and the "
      "sentence and the table agree")
