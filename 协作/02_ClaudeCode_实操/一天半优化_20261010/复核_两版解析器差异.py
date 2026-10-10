#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 `core_v3.py`（原件）与 `core_v3_复核修复.py`（我的副本）在**真实格式**上的行为差异。

场景来自实测：起点修复轮 **360/360** 原答都带 `<think>\\n\\n</think>\\n\\n` 前缀，
而 v2 的适配目标同样是裸 JSON —— 说明"用裸 JSON 训练"并不能去掉这个包装。
所以 v3 的生成几乎必然带前缀；若解析器不剥它，奖励会恒为 0。

做法：拿 v3 适配目标里的**真实答案文本**，分别
  ① 原样（裸 JSON）
  ② 前面加上完整空 think 前缀
喂给两个解析器，比较"能否解析成功"。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

CODEX = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODEX))
import core_v3 as ORIG  # noqa: E402

spec = importlib.util.spec_from_file_location("fix", HERE / "core_v3_复核修复.py")
FIX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FIX)

rows = [json.loads(l) for l in
        (CODEX / "data_v3" / "adapt512_sft.jsonl").open(encoding="utf-8") if l.strip()]
answers = [r["messages"][-1]["content"] for r in rows]


def _ok(parser, text) -> bool:
    try:
        return isinstance(parser(text), dict)
    except Exception:
        return False


def _ang(score_fn, text, ref) -> float:
    try:
        return score_fn(text, ref)["parts"]["angular"]
    except Exception:
        return float("nan")


THINK = "<think>\n\n</think>\n\n"
res = {}
for label, wrap in (("裸 JSON", lambda s: s), ("带完整空 think 前缀", lambda s: THINK + s)):
    o = sum(1 for a in answers if _ok(ORIG.parse, wrap(a)))
    f = sum(1 for a in answers if _ok(FIX.parse, wrap(a)))
    res[label] = {"原件可解析": f"{o}/{len(answers)}",
                  "复核版可解析": f"{f}/{len(answers)}"}

# 扇区分白拿的对照（用真实参考构造"角向答 unknown"的答案）
ref_rows = [json.loads(l) for l in
            (CODEX / "pools" / "rl.jsonl").open(encoding="utf-8") if l.strip()]
ref = next(r["reference"] for r in ref_rows
           if r["reference"]["angular_type"] == "nondirectional")
bad = json.dumps({**ref, "angular_type": "unknown"}, ensure_ascii=False)
res["角向答unknown且参考nondirectional"] = {
    "原件角向分": round(_ang(ORIG.score, bad, ref), 4),
    "复核版角向分": round(_ang(FIX.score, bad, ref), 4),
    "原件总分": round(ORIG.score(bad, ref)["reward"], 4),
    "复核版总分": round(FIX.score(bad, ref)["reward"], 4),
}
res["参考样例"] = {k: ref[k] for k in ("defect_class", "coverage_band",
                                       "red_mass_zones", "angular_type", "clock_sectors")}
print(json.dumps(res, ensure_ascii=False, indent=1))
(HERE / "复核_两版解析器差异.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
