#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 8 模型 × 36 图的**冻结枚举**描述复核结果。

**本脚本不解析自由文本。** 它只统计复核者直接给出的枚举值 ——
这是 Codex 点名禁止用自由文本正则打分之后的**唯一口径**。

六维统一四值：`SUPPORTED`（好）/ `PARTIAL`（部分/轻微）/ `CONTRADICTED`（坏）/
`UNCERTAIN`（判不了）；另有 `MISSING`（字段为空）与 `NEEDS_REVIEW`（归属不明）。
**负向维度的极性已在复核侧统一**：`SUPPORTED` 恒为「好」。

方向**分列三件事**（Codex 要求「没给方向」与「方向答对」分开算）：
  · 该不该有方向：`applicable` ∈ {true, false, unknown}
  · 给了什么：`verdict` ∈ {SUPPORTED, PARTIAL, CONTRADICTED, UNCERTAIN, NO_ANSWER}
  · **不适用 / 未答 / 可评且吻合** 三个数必须分别出现。

跑法：python 汇总枚举复核.py
"""
from __future__ import annotations
import io, json
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
RES = D / "复核结果"
KEY = D / "盲号对照_key.json"
OUT_MD = D / "描述对照_枚举.md"
OUT_JS = D / "描述对照_枚举.json"

DIMS = [("形态", "主要形态是否有图像支持"), ("位置", "位置是否吻合"),
        ("遗漏", "主要结构是否有遗漏"), ("断言", "是否无依据断言"),
        ("格式", "格式约定问题")]
VERDICTS = ["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN", "MISSING",
            "NEEDS_REVIEW"]
DVERDICTS = VERDICTS + ["NO_ANSWER"]


def cell(v, allowed):
    """只认枚举值；写成别的（含自由文本结论）一律记 NEEDS_REVIEW，**不默认放行**。"""
    if isinstance(v, dict):
        v = v.get("verdict")
    if v is None:
        return "MISSING"
    s = str(v).strip().upper()
    return s if s in allowed else "NEEDS_REVIEW"


def extract(d):
    out = []
    for it in d.get("条目", []):
        out.append(it)
    return out


def main():
    key = {k["sample_id"]: k["盲号对照"] for k in
           json.loads(KEY.read_text(encoding="utf-8"))}
    rows, missing = [], []
    for i in range(1, 7):
        p = RES / f"分片{i}_复核.json"
        if not p.exists():
            missing.append(p.name); continue
        got = extract(json.loads(p.read_text(encoding="utf-8")))
        print(f"  分片{i}: {len(got)} 图")
        rows += got
    if missing:
        print(f"  !! 尚缺 {missing}")
    per = defaultdict(lambda: defaultdict(Counter))
    dir_stats = defaultdict(Counter)
    detail = []
    for it in rows:
        sid = it.get("sample_id") or it.get("图号")
        mp = key.get(sid) or {}
        for a in it.get("答案集", []):
            lab = a.get("盲号")
            model = mp.get(lab)
            if model is None:
                print(f"  !! {sid} 盲号 {lab} 映射缺失"); continue
            for short, full in DIMS:
                v = cell(a.get(short) if short in a else a.get(full),
                         ["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN",
                          "MISSING", "NEEDS_REVIEW"])
                per[model][short][v] += 1
                detail.append({"sample_id": sid, "盲号": lab, "模型": model,
                               "维度": short, "判定": v})
            dv = a.get("方向") or {}
            app = dv.get("applicable") if isinstance(dv, dict) else None
            app = ("true" if app is True else "false" if app is False
                   else "unknown" if app == "unknown" else "MISSING")
            vv = cell(dv.get("verdict") if isinstance(dv, dict) else None, DVERDICTS)
            dir_stats[model][f"applicable={app}"] += 1
            dir_stats[model][f"verdict={vv}"] += 1
            detail.append({"sample_id": sid, "盲号": lab, "模型": model,
                           "维度": "方向", "判定": f"{app}/{vv}"})

    OUT_JS.write_text(json.dumps(
        {"口径": {"六维": VERDICTS, "方向": DVERDICTS,
                  "禁止": "不解析自由文本；只统计复核者给出的枚举值",
                  "SUPPORTED": "恒为「好」；负向维度的「无」也是 SUPPORTED",
                  "MISSING": "字段为空，不算正确", "NEEDS_REVIEW": "归属不明，不默认通过",
                  "性质": "模型复核草稿，不是人工 gold"},
         "逐模型逐维度": {m: {d: dict(c) for d, c in v.items()} for m, v in per.items()},
         "方向分列": {m: dict(c) for m, c in dir_stats.items()},
         "逐条": detail}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L = ["# 描述事实对照（**冻结枚举**，8 模型 × 36 图）", "",
         "> **全部是模型复核草稿，不是人工 gold。**",
         "> **本项目已决定不做真人复核（2026-10-08），真人两列永久留空。**", "",
         "## 口径（Codex 点名要求）", "",
         "**不允许用自由文本正则给描述结论打分。** 复核者直接给枚举值：",
         "`SUPPORTED`（好）/ `PARTIAL`（部分/轻微）/ `CONTRADICTED`（坏）/ "
         "`UNCERTAIN`（判不了）/ `MISSING`（字段为空）/ `NEEDS_REVIEW`（归属不明）。",
         "**`SUPPORTED` 恒为「好」**；负向维度（遗漏/断言/格式）的「无」也是 `SUPPORTED`。",
         "**`MISSING` 不算正确，`NEEDS_REVIEW` 不默认通过。**", "",
         "判据先冻结并回归 **20/20** 才允许复核（见 `复核schema_冻结.py`），",
         "堵住上一轮三个缺口：「部分支持」被算干净、无问题词时默认干净、"
         "有结论字段时忽略证据。", "",
         "## 一、六维分布（每模型 36 图 × 5 维 = 180 项）", ""]
    for m in sorted(per):
        L += [f"### `{m}`", "",
              "| 维度 | SUPPORTED | PARTIAL | CONTRADICTED | UNCERTAIN | MISSING | NEEDS_REVIEW |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for short, _ in DIMS:
            c = per[m].get(short, {})
            L.append(f"| {short} | " + " | ".join(str(c.get(v, 0)) for v in VERDICTS) + " |")
        tot = Counter()
        for c in per[m].values():
            tot.update(c)
        L += ["| **合计** | " + " | ".join(f"**{tot.get(v,0)}**" for v in VERDICTS) + " |", ""]
    L += ["## 二、方向：**三件事分列**（Codex 要求）", "",
          "「不适用」「未答」「可评且吻合」必须分开；**「方向干净」不等于「方向答对」**。", "",
          "| 模型 | 适用性分布 | 判定分布 |", "|---|---|---|"]
    for m in sorted(dir_stats):
        c = dir_stats[m]
        app = "；".join(f"{k.split('=')[1]}={v}" for k, v in sorted(c.items())
                       if k.startswith("applicable="))
        ver = "；".join(f"{k.split('=')[1]}={v}" for k, v in sorted(c.items())
                       if k.startswith("verdict="))
        L.append(f"| `{m}` | {app} | {ver} |")
    L += ["", "## 三、边界", "",
          "1. **模型自查**，执行者继承编排会话上下文，**不能替代人工复核**。",
          "2. **36 图是开发集**（val84 每类 2 + dev18，`Near_full` 仅 4），**不是盲测卷**。",
          "3. **六维分项不是独立样本**：36 图 × 5 维 = 180 项**来自 36 张图**，"
          "**不能当 180 个独立观测**；更**不能把某几项的比例说成描述准确率**。",
          "4. **格式不是视觉事实**，与形态/位置的语义不同，不可加总。",
          "5. **外部模型的后端标识与采样参数未知**（Kimi 为原生思考，已在别处单列）。",
          "6. 本表**不含类别正确率** —— 那在 `已验收结果.json` 与报告主表里。"]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n写出 {OUT_MD.name} / {OUT_JS.name}")
    print(f"{'模型':<24}{'SUP':>6}{'PAR':>6}{'CON':>6}{'UNC':>6}{'MIS':>6}{'NR':>6}")
    for m in sorted(per):
        t = Counter()
        for c in per[m].values():
            t.update(c)
        print(f"{m:<24}" + "".join(f"{t.get(v,0):>6}" for v in VERDICTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
