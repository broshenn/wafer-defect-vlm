"""Rewrite LIMITATIONS.md's "how many runs" numbers from the records.

Four places in LIMITATIONS.md carry a count of runs, and every one of them was
hand-typed when it was written:

  8.9   "六个 run 的全部 286 个空转步均满足（逐 run 实测为 54+49+23+57+63+40 ..."
  9     "6 份 RL 记录，每一份 ..."
  9     "当前 6 个 run 全部一致，按该工具 `RUNS` 的顺序：36.00%、32.67%、..."
  9     "六个 run 全部 286 个空转步上的逐值核对 ..."

Three more runs land today. Each landing changes all four, and each one is the kind
of number that stays *syntactically* fine while becoming false -- the count is not
wrong, it is about a run set that no longer exists. The 271-vs-286 line in item 8.9
is the same mistake in its arithmetic form.

So the numbers are read:

  * per-run idle steps and their total come from outputs/reports/kl_length_confound.json
    (which now extends its own run list from disk);
  * the idle-rate list comes from tools/idle_step_table.py's own "in this table's
    order" line -- the order is that tool's decision, not this script's guess;
  * the RL-record count is the number of *_train_result.json records minus SFT.

Every anchor must match exactly once. A missing anchor means the sentence was
reworded and this script would otherwise edit nothing while reporting success --
it exits instead, and writes nothing. --check reports what would change.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = Path(os.environ.get("WAFER_DOC", str(ROOT / "LIMITATIONS.md")))
NUMERAL = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八",
           9: "九", 10: "十", 11: "十一", 12: "十二"}


def numeral(n: int) -> str:
    return NUMERAL.get(n, str(n))


def idle_counts():
    rec = json.loads((ROOT / "outputs/reports/kl_length_confound.json")
                     .read_text(encoding="utf-8"))
    per = [v["kl_identity"]["idle_steps_tested"] for v in rec["runs"].values()]
    return per, sum(per)


def idle_rates():
    """The verified rates, in idle_step_table.py's own order."""
    r = subprocess.run([sys.executable, str(ROOT / "tools/idle_step_table.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"idle_step_table.py failed (rc={r.returncode}); nothing written:\n"
                 f"{r.stdout[-1500:]}\n{r.stderr[-800:]}")
    lines = r.stdout.splitlines()
    for i, line in enumerate(lines):
        if "in this table's order" in line:
            for cand in lines[i + 1:]:
                if cand.startswith("|") and "%" in cand:
                    # each cell keeps its own %: the doc writes one per value
                    return [c.strip() for c in cand.strip("|").split("|")]
            break
    sys.exit("could not find idle_step_table.py's rate line; nothing written")


def rl_records():
    return [p for p in (ROOT / "outputs/reports").glob("*_train_result.json")
            if p.name != "sft_train_result.json"]


def audit_counts():
    """(metric numbers, derived arithmetic) as the audit itself reports them.

    Quoted in section 9 as the current snapshot, next to a warning not to reuse it.
    A number that says "do not reuse me" is still a number a reader will reuse, so
    it is kept current -- the values it passed through stay as hand-written history
    beside it.
    """
    r = subprocess.run([sys.executable, str(ROOT / "tools/audit_report_numbers.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"audit_report_numbers.py failed (rc={r.returncode}); nothing "
                 f"written:\n{r.stdout[-1200:]}\n{r.stderr[-600:]}")
    m = re.search(r"metric numbers\s*:\s*(\d+)", r.stdout)
    d = re.search(r"derived arithmetic\s*:\s*(\d+)", r.stdout)
    if not (m and d):
        sys.exit("could not read the audit's summary lines; nothing written")
    return int(m.group(1)), int(d.group(1))


def run_columns():
    """How many run columns 5.2.5's per-class table has (SFT counts as one).

    Counted from the table itself: the sentence that quotes it ("在七个 run 里最高")
    and the sentence that warns about it ("写此句时为七列") must agree with the table
    that is actually there, and neither can be derived from a list of runs -- the
    table is the only thing that knows which columns were added.
    """
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| 类别（F1）"):
            return len(line.strip().strip("|").split("|")) - 1
    sys.exit("5.2.5's per-class table header was not found; nothing written")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    a = ap.parse_args()

    per, total = idle_counts()
    rates = idle_rates()
    n_rec = len(rl_records())
    s = DOC.read_text(encoding="utf-8")
    edits = []

    def sub(pattern, repl, why):
        nonlocal s
        new, k = re.subn(pattern, repl, s, count=0)
        if k != 1:
            sys.exit(f"anchor for '{why}' matched {k} times, expected 1; "
                     f"nothing written")
        if new != s:
            edits.append(why)
        s = new

    def sub_all(pattern, repl, why):
        """Same, for a phrase that legitimately appears more than once. Requires at
        least one match, and reports how many it rewrote."""
        nonlocal s
        new, k = re.subn(pattern, repl, s, count=0)
        if k < 1:
            sys.exit(f"anchor for '{why}' matched nothing; nothing written")
        if new != s:
            edits.append(f"{why} ({k}x)")
        s = new

    # 8.9 per-run idle counts and their total. The numeral is part of the match:
    # matching only from "个 run ..." would leave the old numeral in place and
    # print "六六 run".
    sub(r"[一二三四五六七八九十]+(个 run 的全部 )\d+"
        r"( 个空转步均满足（逐 run 实测为\s*\n?\s*)[\d+]+(；)",
        f"{numeral(len(per))}个 run 的全部 {total} 个空转步均满足"
        f"（逐 run 实测为\n{'+'.join(str(x) for x in per)}；",
        "8.9: idle-step total and per-run list")
    # 9 record count
    sub(r"\d+( 份 RL 记录)", rf"{n_rec}\1", "9: RL record count")
    # 9 verified rates, in the verifying tool's order
    # This sentence writes its count in Arabic numerals ("6 个 run"), unlike 8.9 and
    # the other section-9 bullet, so the count is written as a digit here. The
    # numeral group tolerates either form on the way in: this line was once written
    # as "当前 六 个 run" by an earlier version of this script, and the anchor has
    # to still match it to repair it.
    sub(r"(当前\s*)[\d一二三四五六七八九十]+(\s*个 run 全部一致，"
        r"按该工具 `RUNS` 的顺序：\s*\n?\s*)[\d.%+、]+(%)",
        f"当前 {len(rates)} 个 run 全部一致，按该工具 `RUNS` 的顺序：\n"
        f"  {'、'.join(rates)}",
        "9: verified idle rates")
    # 9 the per-value check's count and total
    sub(r"[一二三四五六七八九十]+个( run 全部 )\d+( 个空转步上的逐值核对)",
        f"{numeral(len(per))}个 run 全部 {total} 个空转步上的逐值核对",
        "9: per-value check count")

    # 9 the audit's own two numbers, read back from the audit
    n_metrics, n_derived = audit_counts()
    sub(r"(三份文档共 )\d+( 个指标数字)",
        f"三份文档共 {n_metrics} 个指标数字",
        "9: the audit's metric-number count")
    sub(r"(其中 )\d+( 个是已记录数字之间的差值)",
        f"其中 {n_derived} 个是已记录数字之间的差值",
        "9: the audit's derived-arithmetic count")

    # 5.2.5: the column count, the warning that quotes it, and the bullets that use it
    cols = run_columns()
    sub(r"(写此句时为)[一二三四五六七八九十]+(列)",
        f"写此句时为{numeral(cols)}列",
        "5.2.5: how many columns when the note was written")
    sub(r"(下面四句里的\s*\n?\s*「)[一二三四五六七八九十]+(个 run」都必须重核)",
        f"下面四句里的\n「{numeral(cols)}个 run」都必须重核",
        "5.2.5: the count the four bullets below must be re-checked against")
    sub_all(r"(在)[一二三四五六七八九十]+(个 run 里最高)",
            f"在{numeral(cols)}个 run 里最高",
            "5.2.5: two bullets that read an extremum off the table")

    if a.check:
        print(f"per-run idle: {per} (total {total}); rates: {len(rates)}; "
              f"RL records: {n_rec}")
        print("would change: " + (", ".join(edits) if edits else "nothing"))
        return 0

    if not edits:
        print("nothing to change; the four counts already match the records")
        return 0
    if s.count("**") % 2:
        sys.exit("bold markers are odd after the edit; nothing written")
    DOC.write_text(s, encoding="utf-8")
    print(f"updated {len(edits)} sentence(s) in {DOC}:")
    for e in edits:
        print(f"  - {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
