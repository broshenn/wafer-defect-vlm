"""Make FINAL_REPORT.md's run lists and idle-step bullet read what is on disk.

Run 43 (GSPO, G=4, lr 5e-5) landed today. It is in paired_significance.json (8 runs
included) and in run_set.py's NAMES, but three things in tools/final_report.py are
hand-written lists of runs that were never extended:

  1. `gspo_runs` -- three training-summary bullets. Run 43's training summary is
     absent from the report, and so will be runs 41 and 45 when they land.
  2. `gspo_label` + `for gname in ("GSPO_lr5e5", "GSPO_lr1e5", "GSPO_G32")` -- the
     macro-F1-vs-SFT lines. Run 43 gets no line, and the guard above it (`if g_f1 is
     None: continue`) makes the omission look deliberate.
  3. `_lbl` in the paired-test section -- five keys, no GSPO_G4_lr5e5.

None of the three is *wrong* when written; each is a snapshot that a later run
silently falls outside of. That is the defect this document's section 8 keeps
recording, so the fix is the same shape as run_set.py's: enumerate, do not list.

The idle-step bullet is the fourth: it is a literal string carrying 15.33% / 36.00%
and the sentence "归因不成立" (the attribution does not hold). Run 43 makes the
attribution half-true -- at G=4 there is now a genuine single-variable IS-level pair --
so the sentence has to change, and the numbers it quotes are read from the records
from now on so the next run cannot leave them behind.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

# --------------------------------------------------------------- helpers, once
A0 = '''def load(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
'''
A0_NEW = A0 + '''

# ---------------------------------------------------------------- run geometry
# Both tag forms ("qwen35_9b_gspo_g4_lr5e5") and comparison-column forms
# ("GSPO_G4_lr5e5") parse with one rule. The defaults are not arbitrary: the two
# oldest GSPO runs were named before the grid existed, and both are G=8 at the
# paper's lr, which is what the omitted fields mean.
def _geom(name: str):
    """-> (group size, learning rate as written). Defaults G=8, lr 5e-5."""
    t = str(name).replace("qwen35_9b_", "")
    g = re.search(r"[gG](\\d+)", t)
    lr = re.search(r"lr([0-9.eE+-]+)", t)
    return (int(g.group(1)) if g else 8,
            {"1e5": "1e-5"}.get(lr.group(1), lr.group(1)) if lr else "5e-5")


GSPO_NOTE = {
    "gspo_lr1e5": "，与 GRPO 对齐以分离算法与学习率",
    "qwen35_9b_gspo_g32": "，论文设定",
    "qwen35_9b_gspo_g4_lr5e5": "，单变量 IS 层级对照的 sequence 一侧",
    "qwen35_9b_gspo_g32_lr1e5": "，与 G=32 lr 5e-5 构成组大小的学习率对照",
}


def gspo_run_list(reports):
    """Every GSPO training record on disk, in grid order (group size, then lr)."""
    found = []
    for p in reports.glob("*gspo*_train_result.json"):
        tag = p.name[: -len("_train_result.json")]
        g, lr = _geom(tag)
        found.append((g, float(lr), p.name,
                      f"GSPO（G={g}，lr {lr}{GSPO_NOTE.get(tag, '')}）"))
    return [(f, lab) for _, _, f, lab in sorted(found)]


# ------------------------------------------------------------------ idle steps
def idle_frac(reports, tag):
    """(share of idle steps, steps logged) for a run; (None, None) if unrecorded."""
    rec = load(reports / f"{tag}_train_result.json") or {}
    f = rec.get("mean_frac_reward_zero_std")
    return (f, rec.get("steps_logged") or 150) if f is not None else (None, None)


def idle_txt(reports, tag):
    f, n = idle_frac(reports, tag)
    return f"{f:.2%}（{round(f * n)}/{n}）" if f is not None else None
'''

# --------------------------------------------------------- 1. training bullets
A1 = '''    gspo_runs = [
        ("qwen35_9b_gspo_v1_train_result.json", "GSPO（G=8，lr 5e-5）"),
        ("gspo_lr1e5_train_result.json", "GSPO（G=8，lr 1e-5，与 GRPO 对齐以分离算法与学习率）"),
        ("qwen35_9b_gspo_g32_train_result.json", "GSPO（G=32，lr 5e-5，论文设定）"),
    ]
'''
A1_NEW = '''    # Enumerated from the records on disk, not listed here. The literal list this
    # replaces named three runs and stayed at three when a fourth landed, so the
    # report described a smaller run set than it contained and nothing in the
    # produced text could show it.
    gspo_runs = gspo_run_list(reports)
'''

# ------------------------------------------- 2. macro-F1-vs-SFT column loop
A2 = '''    gspo_label = {"GSPO_lr5e5": "GSPO_lr5e5（G=8）",
                  "GSPO_lr1e5": "GSPO_lr1e5（G=8）",
                  "GSPO_G32": "GSPO_G32（G=32，论文设定）"}
'''
A2_NEW = '''    # Which GSPO columns exist is a fact about the comparison table; read it. The
    # hand-written tuple below was the third place a new run could land without
    # being added to it, and the `g_f1 is None` guard made that look deliberate.
    def gspo_cols():
        for name in (metrics.get("classification macro-F1", {}) or {}):
            if not str(name).upper().startswith("GSPO"):
                continue
            g, lr = _geom(name)
            note = "，论文设定" if name == "GSPO_G32" else ""
            yield (g, float(lr), name, f"GSPO(G={g}) lr{lr}{note}")
'''
if s.count(A2) != 1:
    sys.exit(f"A2 anchor appears {s.count(A2)} times")

A3 = '''    for gname in ("GSPO_lr5e5", "GSPO_lr1e5", "GSPO_G32"):
'''
A3_NEW = '''    for _g, _lr, gname, gspo_lab in sorted(gspo_cols()):
'''

A4 = '''        line = (f"- {gspo_label.get(gname, gname)} macro-F1 {pct(g_f1)} vs SFT {pct(sft_f1)}"
'''
A4_NEW = '''        line = (f"- {gspo_lab} macro-F1 {pct(g_f1)} vs SFT {pct(sft_f1)}"
'''

# ------------------------------------------------- 3. paired vs-SFT label map
A5 = '''        _lbl = {"GRPO_G4_lr1e5": "GRPO(G=4) lr1e-5", "GRPO_G4_lr5e5": "GRPO(G=4) lr5e-5",
                "GSPO_G8_lr1e5": "GSPO(G=8) lr1e-5", "GSPO_G8_lr5e5": "GSPO(G=8) lr5e-5",
                "GSPO_G32_lr5e5": "GSPO(G=32) lr5e-5"}
'''
A5_NEW = '''        def _vs_sft_label(k):
            """A record key -> a label. Any suffix the key carries beyond
            algorithm/G/lr (a seed run, say) is kept, so two runs of the same cell
            can never print the same label."""
            alg = "GRPO" if str(k).upper().startswith("GRPO") else "GSPO"
            g, lr = _geom(k)
            rest = re.sub(r"^(GRPO|GSPO)", "", str(k), flags=re.I)
            rest = re.sub(r"^_?G\\d+", "", rest)
            rest = re.sub(r"^_?lr[0-9.eE+-]+", "", rest).strip("_")
            tail = (f"（{rest}）" if rest
                    else "（论文设定）" if k == "GSPO_G32_lr5e5" else "")
            return f"{alg}(G={g}) lr{lr}{tail}"
        # Enumerated from the record, not listed here: run 43 was in
        # comparisons_vs_SFT and had no line in this section.
        _lbl = {k: _vs_sft_label(k) for k in sorted(
            ps["comparisons_vs_SFT"], key=lambda k: (_geom(k)[0], float(_geom(k)[1])))}
'''

# ------------------------------------------------------- 4. idle-step bullet
A6 = '''    out.append(
        "- **空转步的下降是观察到的，但归因不成立，且没有转化为分数**：lr 1e-5 下 "
        "GSPO 的零优势步是 15.33%（23/150），GRPO 是 36.00%（54/150）。"
        "但该对照**同时改了组大小**（G=8 对 G=4），本设计无法把「IS 层级」与"
        "「组大小」分开，所以这是**观察到的差异，不是归因给序列级归一化的证据**"
        "（见 LIMITATIONS 5.2.3）。无论归因如何，它都没有转化为基准分数 —— "
        "优势信号的多寡与最终指标的高低，在这里是解耦的。")
'''
A6_NEW = '''    # Read from the records rather than written here: the previous version carried
    # 15.33% / 36.00% and "归因不成立" inline, so a new run could neither update the
    # numbers nor force the sentence to be reconsidered.
    _il = None
    if _psf.is_file():
        try:
            _il = json.loads(_psf.read_text(encoding="utf-8")).get("is_level_paired")
        except Exception:
            _il = None
    _f_g_lr5, _ = idle_frac(reports, "qwen35_9b_grpo_lr5e5")
    _f_s_g4, _ = idle_frac(reports, "qwen35_9b_gspo_g4_lr5e5")
    _seg = ["- **空转步的下降是观察到的；能归因的那一对确实动了，但仍没有转化为分数**："]
    _t1 = idle_txt(reports, "gspo_lr1e5")
    _t2 = idle_txt(reports, "grpo")
    if _t1 and _t2:
        _seg.append(f"lr 1e-5 下 GSPO 的零优势步是 {_t1}，GRPO 是 {_t2} —— "
                    "但该对照**同时改了组大小**（G=8 对 G=4）与 IS 层级，本设计无法把"
                    "「IS 层级」与「组大小」分开，所以这一条只是观察，不是归因。")
    if _f_g_lr5 is not None and _f_s_g4 is not None:
        _seg.append(f"能归因的那一对是 GRPO(G=4) lr5e-5 "
                    f"{idle_txt(reports, 'qwen35_9b_grpo_lr5e5')} 对 GSPO(G=4) lr5e-5 "
                    f"{idle_txt(reports, 'qwen35_9b_gspo_g4_lr5e5')}：同组大小 4、"
                    "同有效批次 16、同学习率 5e-5、同种子 3407，只差 "
                    "`--importance_sampling_level`（记录里 `single_variable` 为真），"
                    f"空转步低 **{(_f_g_lr5 - _f_s_g4) * 100:.2f} 个百分点**。")
    if _il:
        _seg.append(f"但**同一对配置**的准确率差是 "
                    f"{_il['accuracy_delta_sequence_minus_token']:+.4f}、95% CI "
                    f"[{_il['delta_ci95_rows'][0]:+.4f}, {_il['delta_ci95_rows'][1]:+.4f}]"
                    f"（包含 0，McNemar p = {_il['mcnemar']['p_exact_two_sided']}）——"
                    "机制层面的差异测得出，基准分数上的差异测不出"
                    "（见 LIMITATIONS 5.2.3 与 5.2.6）。")
    else:
        _seg.append("它没有转化为基准分数 —— 优势信号的多寡与最终指标的高低，"
                    "在这里是解耦的。")
    out.append("".join(_seg))
'''

EDITS = [(A0, A0_NEW, "helpers: geometry, enumerated GSPO runs, idle-step readers"),
         (A1, A1_NEW, "training bullets: enumerate the records on disk"),
         (A2, A2_NEW, "macro-F1 lines: read which GSPO columns exist"),
         (A3, A3_NEW, "macro-F1 lines: iterate them in grid order"),
         (A4, A4_NEW, "macro-F1 lines: label from the derived name"),
         (A5, A5_NEW, "paired vs-SFT: enumerate the record's comparisons"),
         (A6, A6_NEW, "idle-step bullet: derived numbers, and the IS pair")]

for old, new, why in EDITS:
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    s = s.replace(old, new)

# Rehearsal: the edited file must still parse, and must still contain the four
# places we touched (a replace that silently ate a neighbouring line would pass
# ast.parse but change behaviour).
try:
    tree = ast.parse(s)
except SyntaxError as e:
    sys.exit(f"edited file does not parse: {e}; nothing written")

for want in ("def gspo_run_list", "def idle_txt", "gspo_cols", "_vs_sft_label",
             "single_variable"):
    if want not in s:
        sys.exit(f"edited file is missing {want!r}; nothing written")

P.write_text(s, encoding="utf-8")
print(f"{len(EDITS)} edit(s) applied to {P.name} "
      f"({len(s.splitlines())} lines, parses, {len(tree.body)} top-level statements)")
for _, _, why in EDITS:
    print(f"  - {why}")
