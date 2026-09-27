"""Close the two script defects section 8 items 10 and 11 defer, and regenerate the table
that item 11's prose was based on.

**Why this waited.** bash reads a script by byte offset as it executes it. Run 42 was
still inside `29_grpo_train.sh` -- its result-writing heredoc is read *after* training --
so editing that file while pid 3699 lived would have had the run resume reading from a
stale offset. This refuses to run while the script is live, which is the same rule stated
as a precondition rather than a comment.

**Item 10's fix is the removal of the default, not a better default.** `RUN_TAG`
defaulted to the bare string `grpo`; the GSPO-v1 run therefore overwrote the earlier GRPO
run's `29_grpo_train.log`. Nothing on disk is wrong today only because no record happens
to reference the clobbered file -- one coincidence deep. Any replacement default has the
same defect as the original, because the set of run names is not knowable from inside the
script: a default that happens to be right is a default that is never noticed. So the line
becomes `${RUN_TAG:?...}`, which fails immediately and says why.

**Item 11's fix reads the environment**, which is where the truth already was: the queue
scripts have always passed `GRAD_ACCUM` (run 41 passed 32, run 42 passed 4), and the
heredoc simply did not read it. The line now mirrors the `num_generations` line three
lines above it, which was written correctly from the start -- the defect was never the
mechanism, only that this one field was a literal.

**The table is regenerated from the logs rather than extended by hand.** Item 11's table
listed five rows and the project now has nine runs; the two it omitted include run 41,
whose record writes `grad_accum: 4` while its own log says
`gradient_accumulation_steps 32` -- the first *wrong* record for a run the document
actually uses. A table enumerating a population has to be regenerated from that
population, and the values in the 「实际累积步」 column are read from each run's own log
by this script, never typed. The label/record/log mapping is taken from
`tools/idle_step_table.py`'s `RUNS`, which is the one place those three are declared
together (and which has its own audit for undeclared logs).

The 「记录写的」 column for a `4 | 4` row says the record is right *by coincidence*,
because that is what it is: the constant equalled the measurement, which is exactly the
condition under which the constant was never going to be noticed.
"""
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
SCRIPT = ROOT / "projects/wafer-defect-vlm/scripts/29_grpo_train.sh"
DOC = ROOT / "LIMITATIONS.md"
IDLE = ROOT / "tools/idle_step_table.py"
REP = ROOT / "outputs/reports"
LOGS = ROOT / "logs"
SB = "/tmp/29_grpo_train.sh.bak-patch118"
DB = "/tmp/LIMITATIONS.md.bak-patch118"
IB = "/tmp/idle_step_table.py.bak-patch118"

# --------------------------------------------------------------- 0. the precondition
live = []
for p in pathlib.Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if "29_grpo_train.sh" in cmd or "queue_grpo_seed2" in cmd:
        live.append((p.name, cmd.strip()[:80]))
if live:
    sys.exit("a process is still reading 29_grpo_train.sh, so it will not be edited:\n"
             + "\n".join(f"  pid {a}: {b}" for a, b in live)
             + "\n  bash reads a script by byte offset as it executes it; editing it now "
               "would have that run resume mid-line. Re-run this after it exits.")

# ------------------------------------------------------------- 1. the two script fixes
s = SCRIPT.read_text(encoding="utf-8")
shutil.copy2(SCRIPT, SB)

FIXES = [
    ('RUN_TAG="${RUN_TAG:-grpo}"',
     '# No default. A bare `grpo` default let the GSPO-v1 run overwrite an earlier GRPO\n'
     '# run\'s `29_grpo_train.log`: nothing on disk was wrong only because no record\n'
     '# referenced the clobbered file, which is one coincidence deep. A better default\n'
     '# would have the same defect -- the set of run names is not knowable from inside\n'
     '# this script -- so the absence is made explicit instead. Every queue already\n'
     '# passes RUN_TAG; a missing one now fails here, loudly, with this reason.\n'
     'RUN_TAG="${RUN_TAG:?RUN_TAG must be set explicitly: each run needs its own log '
     'and result name, and a default silently shares them between runs}"'),
    ('        "grad_accum": 4,',
     '        # Was the literal 4. The value was never knowable from the literal, so the\n'
     '        # field could not disagree with reality -- it simply did not describe it.\n'
     '        # The queue scripts always passed GRAD_ACCUM (run 41 passed 32, run 42\n'
     '        # passed 4); this line now reads it, as `num_generations` above always did.\n'
     '        "grad_accum": int(os.environ.get("GRAD_ACCUM", 4)),'),
]
if "grad_accum_source" in s:
    print("29_grpo_train.sh: already fixed by tools/patch75_grpo_record.py -- its "
          "`grad_accum_source` field is present, so this section stands down and only the "
          "table and the guards below run. patch75 fixes the same two lines and is the "
          "fuller fix of the two (it also records where the value came from), and it is "
          "run by 44_final_consolidation.sh step 7 as well as by this landing. Two patches "
          "editing one line is not redundancy but a race, and the loser exits with "
          "'anchor appears 0 times', which leaves a reader unsure which fix is in.")
else:
    for old, new in FIXES:
        if s.count(old) != 1:
            sys.exit(f"the script anchor {old[:44]!r} matches {s.count(old)} times; "
                     f"nothing written")
        s = s.replace(old, new, 1)
    SCRIPT.write_text(s, encoding="utf-8")
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    if r.returncode != 0:
        shutil.copy2(SB, SCRIPT)
        sys.exit(f"the patched script fails `bash -n`; restored:\n{r.stderr}")
    print("29_grpo_train.sh: RUN_TAG now has no default, and grad_accum reads GRAD_ACCUM.")

# Checked here, unconditionally, rather than only inside the branch that
# writes the file. The stand-down above fires whenever patch75 has run first --
# which is the normal order -- so the `bash -n` in the else branch cannot see a
# syntax error introduced by patch75. On 2026-09-16 exactly that happened: the
# one check that would have caught it was unreachable by construction.
_syn = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True,
                      text=True)
if _syn.returncode != 0:
    sys.exit("29_grpo_train.sh does not parse; something else wrote it:\n"
             + _syn.stderr)

# --------------------------------------------- 2. the item-11 table, from the logs
entries = []
for node in ast.parse(IDLE.read_text(encoding="utf-8")).body:
    if (isinstance(node, ast.Assign) and node.targets
            and getattr(node.targets[0], "id", None) == "RUNS"):
        for e in node.value.elts:
            lab, stem, logs = ast.literal_eval(e)
            entries.append((lab, stem if stem.endswith(".json") else stem + ".json",
                            list(logs)))
if not entries:
    shutil.copy2(SB, SCRIPT)
    sys.exit("no RUNS list found in tools/idle_step_table.py; nothing written")

rows, missing, wrong = [], [], []
for lab, stem, logs in entries:
    real = None
    for lg in logs:
        f = LOGS / lg
        if not f.is_file():
            continue
        # The swift args dump: `gradient_accumulation_steps 32` or `...=32`.
        m = re.findall(r"(?<![A-Za-z0-9_])gradient_accumulation_steps=?(\d+)",
                       f.read_text(encoding="utf-8", errors="replace"))
        if m:
            real = int(m[-1])
            break
    rp = REP / stem
    written = None
    if rp.is_file():
        written = (json.loads(rp.read_text(encoding="utf-8")).get("config") or {}
                   ).get("grad_accum")
    if real is None or written is None:
        missing.append(lab)
        continue
    if real != written:
        wrong.append(lab)
    rows.append((lab, real, written))

def verdict(real, written):
    """A `4 | 4` row is right *by coincidence*: the literal equalled the measurement,
    which is precisely the case in which a hard-coded value is never noticed."""
    if real == written:
        return "对（常量碰巧等于实测值）" if real == 4 else "对"
    return "**错**"

body = "".join(f"    | {lab} | {real} | {written} | {verdict(real, written)} |\n"
               for lab, real, written in rows)
table = ("    | run | 实际累积步（该 run 自己的日志）| 记录写的 | |\n"
         "    | --- | --- | --- | --- |\n" + body)
if missing:
    table += f"    （{len(missing)} 个 run 的日志或记录缺失，未列入：{'、'.join(missing)}）\n"

# ------------------------------------------------------- 3. item 11's prose, and item 10
doc = DOC.read_text(encoding="utf-8")
shutil.copy2(DOC, DB)

P11_OLD = """    `scripts/29_grpo_train.sh` 写结果的 heredoc 里，`"grad_accum": 4` 是**字面量**，
    不读任何环境变量。所以它不可能与实际不符 —— 它根本不知道实际是什么。
    训练日志里的 swift 参数才是实测值："""
P11_NEW = """    `scripts/29_grpo_train.sh` 写结果的 heredoc 里，`"grad_accum": 4` 是**字面量**，
    不读任何环境变量。所以它不可能与实际不符 —— 它根本不知道实际是什么。
    **已改为读 `GRAD_ACCUM`**（训练结束后才改，因为队列 42 当时正在读该脚本）；
    队列脚本一直在传这个变量（run 41 传 32、run 42 传 4），只有 heredoc 没读它 ——
    同一段里三行之上的 `num_generations` 从第一天起就是对的，所以错的从来不是做法，
    只是这一个字段被写成了常量。**下表由该脚本从每个 run 自己的日志重新生成**，
    不再手工维护：原先的五行漏掉了 run 41，而 run 41 的记录恰好是**错的**
    （写 4、实际 32）。训练日志里的 swift 参数才是实测值："""
if doc.count(P11_OLD) != 1:
    shutil.copy2(DB, DOC); shutil.copy2(SB, SCRIPT)
    sys.exit(f"item 11's prose anchors {doc.count(P11_OLD)} times; nothing written")
doc = doc.replace(P11_OLD, P11_NEW, 1)

P11_T_OLD = re.search(r"    \| run \| 实际累积步 \| 记录写的 \|\n(?:\s*\|.*\n){4,8}", doc)
if not P11_T_OLD:
    shutil.copy2(DB, DOC); shutil.copy2(SB, SCRIPT)
    sys.exit("item 11's table header did not anchor; nothing written")
doc = doc.replace(P11_T_OLD.group(0), table, 1)

P10_OLD = """    - 一处**尚未发作**的隐患：`scripts/29_grpo_train.sh` 的"""
P10_NEW = """    - 一处**尚未发作的隐患已修**：`scripts/29_grpo_train.sh` 的"""
if doc.count(P10_OLD) != 1:
    shutil.copy2(DB, DOC); shutil.copy2(SB, SCRIPT)
    sys.exit("item 10's opening anchors != 1; nothing written")
doc = doc.replace(P10_OLD, P10_NEW, 1)

P10B_OLD = """      今天两个队列都显式传了 `RUN_TAG`，不会复发；
      默认值本身的修复留到收盘后（见第 8 条：不得在 run 正在读脚本时改它）。"""
P10B_NEW = """      修复是**取消默认值**而不是换一个默认值：该行现在是 `RUN_TAG="${RUN_TAG:?…}"`，
      漏传时脚本立即失败并打印原因。没有「正确的默认值」可选 —— 任何默认值都会让两个
      run 争用同一个日志名，而默认值恰好正确的那些情况，正是它永远不会被发现的情况。
      （修复在队列 42 的进程退出后进行：此前该脚本正被它按字节偏移读取，
      见本节第一条。两个队列都显式传了 `RUN_TAG`，所以期间没有 run 受影响。）"""
if doc.count(P10B_OLD) != 1:
    shutil.copy2(DB, DOC); shutil.copy2(SB, SCRIPT)
    sys.exit("item 10's deferral sentence anchors != 1; nothing written")
doc = doc.replace(P10B_OLD, P10B_NEW, 1)
DOC.write_text(doc, encoding="utf-8")
print(f"LIMITATIONS.md: items 10 and 11 closed; the table regenerated over "
      f"{len(rows)} run(s), {len(wrong)} of them wrong"
      + (f" ({', '.join(wrong)})" if wrong else ""))

# --------------------- 3b. the table check, in the shape this patch creates
# The item lives here rather than in the checker because it reads the table this patch
# regenerates: `| run | 实际累积步（该 run 自己的日志）| 记录写的 | |`. Written against the
# old three-column shape it would be deleted by this patch; written against the new one
# and placed before this patch runs, it is a checker that is red until a patch that
# cannot run yet (pid 3699 is still reading the script). A guard belongs with the
# artifact it guards.
C_BLOCK = '''
# ------------------------- 11: the table of runs whose record disagrees with its log
# A table of defects is a population, and run 41 was missing from it while its own
# record disagreed with its own log (4 written, 32 logged). Three columns are read back
# against the runs: the labels, the accumulations the logs show, the values the records
# write, and which rows are marked wrong -- the mark and the disagreement have to be the
# same set, or the mark is a claim about the log that nothing computes.
P_I11 = (r"\\| run \\| 实际累积步（该 run 自己的日志）\\| 记录写的 \\| \\|\\s*\\n"
         r"\\| --- \\| --- \\| --- \\| --- \\|\\s*\\n"
         r"((?:\\s*\\|[^|\\n]*G=\\d+[^|\\n]*\\|[^\\n]*\\n)+)")
g = anchor("11: the table of runs whose record disagrees with its own log", P_I11)
if g:
    _decl = {}
    if _IDLE_TOOL.is_file():
        for _node in ast.parse(_IDLE_TOOL.read_text(encoding="utf-8")).body:
            if (isinstance(_node, ast.Assign) and _node.targets
                    and getattr(_node.targets[0], "id", None) == "RUNS"):
                for _e in _node.value.elts:
                    _lab, _s, _lg = ast.literal_eval(_e)
                    _decl[_lab] = (_s if _s.endswith(".json") else _s + ".json", list(_lg))
    _rows, _rec, _gacc = {}, {}, {}
    for _ln in g.group(1).splitlines():
        _c = [x.strip() for x in _ln.strip().strip("|").split("|")]
        if len(_c) >= 4 and re.search(r"G=\\d+", _c[0]):
            _rows[_c[0]] = (_c[1], _c[2].strip("*").strip(), "错" in _c[3])
    for _lab, (_stem, _logs) in sorted(_decl.items()):
        _f = REP / _stem
        if _f.is_file():
            _w = (json.loads(_f.read_text(encoding="utf-8")).get("config") or {}).get(
                "grad_accum")
            if _w is not None:
                _rec[_lab] = int(_w)
        for _lg in _logs:
            _lf = ROOT / "logs" / _lg
            if _lf.is_file():
                _m = re.findall(r"(?<![A-Za-z0-9_])gradient_accumulation_steps=?(\\d+)",
                                _lf.read_text(encoding="utf-8", errors="replace"))
                if _m:
                    _gacc[_lab] = int(_m[-1])
                    break
    cmp("11: the runs the table lists, against the runs that have a record",
        sorted(_rows), sorted(_rec),
        why="a run missing from a table of wrong records is a record vouched for by "
            "omission; run 41 was the instance")
    cmp("11: the accumulations it prints from the logs",
        {k: int(v[0]) for k, v in _rows.items() if v[0].isdigit()},
        {k: _gacc[k] for k in _rows if k in _gacc})
    cmp("11: the accumulations it prints for the records",
        {k: int(v[1]) for k, v in _rows.items() if v[1].isdigit()},
        {k: _rec[k] for k in _rows if k in _rec})
    cmp("11: the runs it marks as wrong",
        sorted(k for k, v in _rows.items() if v[2]),
        sorted(k for k in _rows if k in _rec and k in _gacc and _rec[k] != _gacc[k]),
        why="the mark has to be computed from the log; a mark beside two numbers is a "
            "reader's inference printed as a check")
'''
CHK = ROOT / "tools/check_quantified_claims.py"
chk = CHK.read_text(encoding="utf-8")
shutil.copy2(CHK, "/tmp/check_quantified_claims.py.bak-p118b")
MARK = "the idle-step rate, its worst run, and its population"
_lines = chk.splitlines(keepends=True)
_hits = [i for i, ln in enumerate(_lines) if MARK in ln]
if len(_hits) != 1:
    sys.exit(f"the checker insertion line matches {len(_hits)} time(s); nothing written")
_lines.insert(_hits[0], C_BLOCK.lstrip("\n") + "\n")
CHK.write_text("".join(_lines), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(CHK)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-p118b", CHK)
    sys.exit(f"the patched checker does not compile; restored:\n{r.stderr}")
print("the checker now reads item 11's table: its runs, both columns, and its marks.")

# ------------------------------------- 4. the undeclared log, into the tool that flags it
i = IDLE.read_text(encoding="utf-8")
shutil.copy2(IDLE, IB)
L_OLD = '''    ("GRPO G=4 lr1e-5 seed3408", "qwen35_9b_grpo_lr1e5_seed3408_train_result.json",
     ["24_grpo_qwen35_9b_grpo_lr1e5_seed3408.log"]),'''
L_NEW = '''    ("GRPO G=4 lr1e-5 seed3408", "qwen35_9b_grpo_lr1e5_seed3408_train_result.json",
     ["24_grpo_qwen35_9b_grpo_lr1e5_seed3408.log",
      "29_qwen35_9b_grpo_lr1e5_seed3408_train.log"]),'''
if i.count(L_OLD) != 1:
    shutil.copy2(DB, DOC); shutil.copy2(SB, SCRIPT)
    sys.exit(f"the seed3408 RUNS entry anchors {i.count(L_OLD)} times; nothing written")
IDLE.write_text(i.replace(L_OLD, L_NEW, 1), encoding="utf-8")
print("idle_step_table.py: run 42's second log is declared, so the audit stops "
      "reporting it as unaccounted.")

# ------------------------------------------------------------- 5. confirm both guards
for tool in ("tools/check_quantified_claims.py", "tools/idle_step_table.py",
             "tools/audit_log_provenance.py"):
    if not (ROOT / tool).is_file():
        print(f"  (skipping {tool}: not present)")
        continue
    r = subprocess.run([sys.executable, str(ROOT / tool)], capture_output=True,
                       text=True, cwd=str(ROOT))
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-3:]
    print(f"\n--- {tool} (exit {r.returncode}) ---")
    for l in tail:
        print("  " + l.strip()[:160])
