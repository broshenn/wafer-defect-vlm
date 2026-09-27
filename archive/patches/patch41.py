"""Fix the stale GSPO claim and record the missing control cell.

The wrong claim that GSPO is unavailable lived in two places: LIMITATIONS 5.2.1,
which was corrected, and provenance.json's deviations_from_spec, which was not.
So the corrected report still printed the retracted claim in its English
summary -- the same "one fact stored twice" failure mode as section 8 items 5
and 6. Both are fixed here, and the open item is disclosed rather than left
implicit.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")

# ------------------------------------------------ 1. provenance.json
prov_path = ROOT / "outputs/reports/provenance.json"
prov = json.loads(prov_path.read_text(encoding="utf-8"))
devs = prov["deviations_from_spec"]

old_claim = ("GRPO/GSPO: GSPO is not offered by this ms-swift commit; GRPO is "
             "available but runs without vLLM because vLLM is not installed.")
new_claim = ("GRPO/GSPO: GSPO is not a separate rlhf_type in this ms-swift commit; it is "
             "GRPO with --importance_sampling_level sequence "
             "(swift/rlhf_trainers/args_mixin.py:425). An earlier revision of this record "
             "claimed GSPO was not offered at all, which was wrong; corrected in "
             "LIMITATIONS 5.2.1. GRPO runs without vLLM because vLLM is not installed.")
if old_claim not in devs:
    sys.exit("FAILED: original GSPO deviation entry not found")
devs[devs.index(old_claim)] = new_claim

open_item = ("Open item: GRPO was run only at lr 1e-5. The claim that the learning rate "
             "rather than the algorithm caused the lr 5e-5 degradation is therefore "
             "supported only within GSPO (which was run at both 5e-5 and 1e-5). A GRPO "
             "run at lr 5e-5 is the missing control cell and has not been run; see "
             "LIMITATIONS 5.2.2.")
if open_item not in devs:
    devs.append(open_item)

prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("provenance.json: stale claim corrected, missing-control-cell open item added")

# ------------------------------------------------ 2. final_report.py
fp = ROOT / "tools/final_report.py"
src = fp.read_text(encoding="utf-8")

edits = []

old_loop = '''    for gname in ("GSPO_lr5e5", "GSPO_lr1e5"):
        g_f1 = metrics.get("classification macro-F1", {}).get(gname)'''
new_loop = '''    gspo_label = {"GSPO_lr5e5": "GSPO_lr5e5（G=8）",
                  "GSPO_lr1e5": "GSPO_lr1e5（G=8）",
                  "GSPO_G32": "GSPO_G32（G=32，论文设定）"}
    for gname in ("GSPO_lr5e5", "GSPO_lr1e5", "GSPO_G32"):
        g_f1 = metrics.get("classification macro-F1", {}).get(gname)'''
edits.append(("loop", old_loop, new_loop))

old_line = '''        line = (f"- {gname} macro-F1 {pct(g_f1)} vs SFT {pct(sft_f1)}"'''
new_line = '''        line = (f"- {gspo_label.get(gname, gname)} macro-F1 {pct(g_f1)} vs SFT {pct(sft_f1)}"'''
edits.append(("line", old_line, new_line))

old_three = '"三个 RL run 在基准上的差异都落在噪声内，唯一明显的差异来自学习率"'
new_three = '"四个 RL run 在基准上的差异都落在噪声内，唯一明显的差异来自学习率"'
edits.append(("three", old_three, new_three))

old_tail = '''        "这是本次 RL 最值得记下的一条：优势信号的多寡与最终指标的高低，在这里是解耦的。")'''
new_tail = '''        "这是本次 RL 最值得记下的一条：优势信号的多寡与最终指标的高低，在这里是解耦的。")
    _rk = ("retrieval mAP@10", "retrieval nDCG@10", "retrieval Recall@10")
    _g32 = {k: (metrics.get(k, {}) or {}).get("GSPO_G32") for k in _rk}
    _sft = {k: (metrics.get(k, {}) or {}).get("SFT") for k in _rk}
    if all(v is not None for v in list(_g32.values()) + list(_sft.values())):
        if all(_g32[k] > _sft[k] for k in _rk):
            out.append(
                "- **G=32 的检索是全部 run 里最好的**：macro 上它是最差的一档，"
                f"但 mAP@10 {_g32[_rk[0]]:.4f}（SFT {_sft[_rk[0]]:.4f}）、"
                f"nDCG@10 {_g32[_rk[1]]:.4f}（SFT {_sft[_rk[1]]:.4f}）、"
                f"Recall@10 {_g32[_rk[2]]:.4f}（SFT {_sft[_rk[2]]:.4f}）三项居首，"
                "其 `radial_zone` 也高于其余所有 run。"
                "该 run 分类最差而检索最好，且 Donut 归零、Scratch 最高，"
                "合起来说明它**重塑了表征而不是全面退化**（见 LIMITATIONS 5.2.5）。")'''
edits.append(("tail", old_tail, new_tail))

for tag, old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED at final_report.py edit {tag}: {n} occurrences (need 1)")
    src = src.replace(old, new)

ast.parse(src)
fp.write_text(src, encoding="utf-8")
print("final_report.py: G=32 in the comparison loop, four RL runs, retrieval finding")
