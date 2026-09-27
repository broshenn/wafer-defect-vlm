"""Stop the generator asserting extremes it does not check.

The G=32 retrieval block guards on `GSPO_G32 > SFT` for all three metrics, then
asserts three further things: that G=32 leads all three across runs, that it is
the worst run at classification, and hence that it "reshaped the representation
rather than degrading across the board". Only the guard was checked. A fifth run
landed, and of the four assertions two became false while the guard still passed:

  * nDCG@10 and Recall@10 are both led by GRPO(G=4) lr5e-5 (0.3917 / 0.2022
    against G=32's 0.3906 / 0.2008). Only mAP@10 is G=32's.
  * the worst RL run on macro-F1 is GSPO(G=8) lr5e-5 at 0.5014, not G=32 (0.5403).

The numbers were all readable from the table. The defect is that the sentence
hardcoded a relation between them -- the same class as item 9 of section 8, in
generated prose rather than hand-written prose. So compute the relation.
"""
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = p.read_text(encoding="utf-8")

OLD = '''    _rk = ("retrieval mAP@10", "retrieval nDCG@10", "retrieval Recall@10")
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

NEW = '''    _rk = ("retrieval mAP@10", "retrieval nDCG@10", "retrieval Recall@10")
    _g32 = {k: (metrics.get(k, {}) or {}).get("GSPO_G32") for k in _rk}
    _sft = {k: (metrics.get(k, {}) or {}).get("SFT") for k in _rk}
    if all(v is not None for v in list(_g32.values()) + list(_sft.values())):
        # Which run leads each metric, and which RL run is worst on macro-F1,
        # are facts about the table -- read them, do not assert them. An earlier
        # version asserted "all three lead" and "worst at classification" behind
        # a guard that only checked G=32 > SFT, so when the fifth run landed two
        # of those sentences were false and the guard still passed.
        def _best(key):
            row = {n: v for n, v in (metrics.get(key, {}) or {}).items()
                   if isinstance(v, (int, float)) and n != "Base"}
            return (max(row, key=lambda n: row[n]), row) if row else (None, {})

        _lead = {k: _best(k)[0] for k in _rk}
        _f1row = {n: v for n, v in (metrics.get("classification macro-F1", {}) or {}).items()
                  if isinstance(v, (int, float)) and n not in ("Base", "SFT")}
        _worst_f1 = min(_f1row, key=lambda n: _f1row[n]) if _f1row else None
        _g32f1 = (metrics.get("classification macro-F1", {}) or {}).get("GSPO_G32")

        if all(_lead[k] == "GSPO_G32" for k in _rk):
            out.append("- **G=32 的检索在全部 run 里居首**："
                       f"mAP@10 {_g32[_rk[0]]:.4f}、nDCG@10 {_g32[_rk[1]]:.4f}、"
                       f"Recall@10 {_g32[_rk[2]]:.4f}"
                       f"（SFT 分别为 {_sft[_rk[0]]:.4f}、{_sft[_rk[1]]:.4f}、"
                       f"{_sft[_rk[2]]:.4f}）。")
        else:
            _detail = "；".join(
                f"{k.split()[-1]} G=32 {_g32[k]:.4f} 对最高 {_lead[k]} "
                f"{(metrics.get(k, {}) or {}).get(_lead[k]):.4f}" for k in _rk)
            out.append("- **检索指标并非由同一个 run 领先，故「G=32 检索最好」已不成立**："
                       + _detail
                       + f"。G=32 三项都高于 SFT（mAP@10 {_g32[_rk[0]]:.4f} 对 "
                       f"{_sft[_rk[0]]:.4f}），")
            out.append("  但**没有一项是全部 run 最高**，"
                       "因此它不能再被用来支持「G=32 重塑了表征」这一说法"
                       "（见 LIMITATIONS 5.2.5）。"
                       "该说法是本轮被实测推翻的又一处「点估计极值当作结论」。")

        if _worst_f1 == "GSPO_G32":
            out.append(f"- G=32 的 macro-F1 {_g32f1:.4f} 是全部 RL run 里最低的一档，"
                       "**分类最差而检索最好**这一对照成立，与 Donut 归零、"
                       "Scratch 最高合起来说明它**重塑了表征而非全面退化**"
                       "（见 LIMITATIONS 5.2.5）。")
        else:
            out.append(f"- **分类最差的 RL run 不是 G=32 而是 {_worst_f1}**"
                       f"（macro-F1 {_f1row[_worst_f1]:.4f} 对 G=32 的 {_g32f1:.4f}），"
                       "所以「G=32 分类最差而检索最好」这一对照不成立。"
                       "G=32 的表征重塑只能由它自身的画像支持（Donut 归零、"
                       "Scratch 最高、`radial_zone` 全部 run 最高），"
                       "不能再借「分类最差」这个极值（见 LIMITATIONS 5.2.5）。")'''

if s.count(OLD) != 1:
    sys.exit(f"FAILED: the G=32 block appears {s.count(OLD)} times")
p.write_text(s.replace(OLD, NEW), encoding="utf-8")
print("final_report.py: G=32 retrieval block now computes its extremes")
