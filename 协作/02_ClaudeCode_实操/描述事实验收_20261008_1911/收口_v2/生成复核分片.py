#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Codex 的匿名审阅材料**再降一次噪**，切成 3 份互不重叠的复核分片。

给复核子agent看的东西必须**只够判「描述是否符合图」**，
不能让它顺手去判类别 —— 否则又会退化成「用类别正确代替描述正确」。

因此从 `匿名审阅材料.json` 里**再剥掉**：
  · `defect_class`（模型自己的类别预测 —— 留着会诱导按类别打分）
  · 任何模型名、盲号→模型映射
  · 任何公开标签
只留：图、盲号、以及四个**描述性**字段
（morphology / radial_zone / clock_direction / caption_zh）。

跑法：python 生成复核分片.py
"""
from __future__ import annotations
import io, json
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # …/收口_v2
ROOT = HERE.parent
SRC = (ROOT.parent.parent / "01_Codex_指挥" / "验收_20261008_描述事实收口"
       / "审阅副本_v2")
OUT = HERE / "复核分片"
KEEP = ["morphology", "radial_zone", "clock_direction", "caption_zh"]
N_SHARD = 3


def main() -> int:
    mat = json.loads(io.open(SRC / "匿名审阅材料.json", encoding="utf-8").read())
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(mat)
    per = (n + N_SHARD - 1) // N_SHARD
    print(f"总 {n} 图 → {N_SHARD} 片，每片 ≤{per} 图")

    for k in range(N_SHARD):
        chunk = mat[k * per:(k + 1) * per]
        if not chunk:
            continue
        items = []
        for it in chunk:
            ans = []
            for a in it["answers"]:
                f = a.get("fields") or {}
                ans.append({"盲号": a["blind"],
                            "展示口径": a.get("display_mode", ""),
                            **{fld: f.get(fld) for fld in KEEP}})
            items.append({"图号": it["number"],
                          "sample_id": it["sample_id"],
                          "图片": str((SRC / "图" / f"{it['sample_id']}.png").resolve()),
                          "答案集": ans})
        p = OUT / f"分片{k+1}_共{len(items)}图.json"
        io.open(p, "w", encoding="utf-8", newline="\n").write(
            json.dumps({"说明": "描述事实复核材料（只含描述性字段；无类别、无模型名、无标签）",
                        "图数": len(items), "条目": items},
                       ensure_ascii=False, indent=2) + "\n")
        print(f"  写出 {p.name}  {len(items)} 图 / {sum(len(i['答案集']) for i in items)} 答案集")

    # 自检：三片必须互不重叠且覆盖全部
    nums = [i["图号"] for k in range(N_SHARD)
            for i in json.loads(io.open(OUT / f"分片{k+1}_共{len(mat[k*per:(k+1)*per])}图.json",
                                        encoding="utf-8").read())["条目"]]
    assert len(nums) == len(set(nums)) == n, (len(nums), len(set(nums)), n)
    print(f"  自检通过：{n} 图，无重叠，全覆盖")
    print(f"  自检：材料里不含 defect_class -> "
          f"{'defect_class' not in json.dumps(json.loads(io.open(OUT / '分片1_共12图.json', encoding='utf-8').read()), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
