#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Phase D 的 7 批 AI 复核结果（只统计枚举值，不解析自由文本）。

- 逐条先规范化，**自检总数必须 = 56 图 × 3 模型 = 168**；
- 通过 `AI复核_盲号对照_key.json`（seed=3407）还原「甲/乙/丙 → 模型」；
- 视觉四维（形态/位置/遗漏/断言）与**格式分列**，方向另立三列；
- `SUPPORTED` 恒为好；`MISSING`/`NEEDS_REVIEW` 不默认通过。
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
RES = D / "AI复核结果"
KEY = D / "AI复核_盲号对照_key.json"
DIMS = ["形态", "位置", "遗漏", "断言"]
VERDICTS = ["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN", "MISSING", "NEEDS_REVIEW"]
DVERDICTS = VERDICTS + ["NO_ANSWER"]


def main() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))          # {sid: {甲:model,...}}
    rows, notes = [], []
    for b in range(1, 8):
        p = RES / f"批次{b}_复核.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["逐条"]:
            rows.append(r)
        notes.append(d.get("方向规则", ""))
    assert len(rows) == 168, f"应 168 条（56 图 × 3 模型），实际 {len(rows)}"
    print(f"规范化后 {len(rows)} 条 ✓（56 图 × 3 模型）")

    # 反查 item_id → sample_id
    fz = json.loads((D / "冻结清单.json").read_text(encoding="utf-8"))
    item2sid = {r["item_id"]: r["sample_id"] for r in fz["逐图"]}

    per = defaultdict(lambda: defaultdict(Counter))
    dirs = defaultdict(Counter)
    seen = set()
    detail = []
    for r in rows:
        sid = item2sid[r["item_id"]]
        model = key[sid][r["盲号"]]
        k = (r["item_id"], r["盲号"])
        assert k not in seen, f"重复 {k}"
        seen.add(k)
        for dim in DIMS:
            v = str(r[dim]).strip().upper()
            assert v in VERDICTS, f"{k} {dim} 非法值 {v}"
            per[model][dim][v] += 1
            detail.append({"item_id": r["item_id"], "模型": model, "维度": dim, "判定": v,
                           "证据": r.get("证据", "")})
        dv = r["方向"]
        app = dv["applicable"] if dv["applicable"] in (True, False, "unknown") else "unknown"
        ver = str(dv["verdict"]).strip().upper()
        assert ver in DVERDICTS, f"{k} 方向非法 {ver}"
        dirs[model][f"applicable={app}"] += 1
        dirs[model][f"verdict={ver}"] += 1
        detail.append({"item_id": r["item_id"], "模型": model, "维度": "方向",
                       "判定": f"{app}/{ver}", "证据": r.get("证据", "")})
        v = str(r["格式"]).strip().upper()
        assert v in VERDICTS, f"{k} 格式非法 {v}"
        per[model]["格式"][v] += 1
        detail.append({"item_id": r["item_id"], "模型": model, "维度": "格式", "判定": v,
                       "证据": r.get("证据", "")})

    out = {"性质": "**模型复核（AI 自查），不是人工 gold**；复核者为执行会话本人，"
                   "先看原图写下 图像事实_预读.json，再对匿名编号回答逐项打分。",
           "上下文限制": ["复核者亲自跑过这三个模型，知道候选集合 → 不是严格盲评；",
                          "旧 AI 意见（Codex_D36 等）分歧原样保留，未做平均；",
                          "本 56 图没有外部大模型原答，不与旧 36 图外部分数比较。"],
           "口径": {"五维枚举": VERDICTS, "方向": DVERDICTS,
                    "SUPPORTED": "恒为「好」；负向维度（遗漏/断言/格式）的「无」也是 SUPPORTED",
                    "MISSING": "不计为正确", "NEEDS_REVIEW": "不默认通过",
                    "分列": "视觉四维（形态/位置/遗漏/断言）与格式**分列**；都不合成单一「描述准确率」",
                    "方向规则": notes},
           "逐模型逐维度": {m: {d: dict(c) for d, c in v.items()} for m, v in per.items()},
           "方向分列": {m: dict(c) for m, c in dirs.items()},
           "逐条": detail}
    (D / "AI描述验收.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")

    L = ["# AI 描述验收（56 图 × 3 模型 = 168 条）", "",
         "> **模型复核（AI 自查），不是人工 gold。** 复核者：Claude Code 执行会话本人。",
         "> 先看原图写下预读事实（`图像事实_预读.json`，早于任何模型原答），",
         "> 再对**匿名编号**（逐图洗牌，seed 3407）的回答逐项打冻结枚举。",
         "> **不是严格盲评** —— 复核者亲自跑了这三个模型。", "",
         "## 一、视觉四维（形态 / 位置 / 遗漏 / 断言；每模型 56×4 = 224 项）", "",
         "| 模型 | SUPPORTED 好 | PARTIAL | CONTRADICTED 坏 | 其它 |", "|---|---:|---:|---:|---:|"]
    for m in ["Base", "L-N3072-3407", "D-N3072-3407"]:
        t = Counter()
        for d in DIMS:
            t.update(per[m][d])
        other = t.get("UNCERTAIN", 0) + t.get("MISSING", 0) + t.get("NEEDS_REVIEW", 0)
        L.append(f"| `{m}` | {t.get('SUPPORTED',0)} | {t.get('PARTIAL',0)} | "
                 f"{t.get('CONTRADICTED',0)} | {other} |")
    L += ["", "## 二、格式（每模型 56 项，**与视觉事实分列**）", "",
          "| 模型 | SUPPORTED 好 | PARTIAL | CONTRADICTED 坏 |", "|---|---:|---:|---:|"]
    for m in ["Base", "L-N3072-3407", "D-N3072-3407"]:
        c = per[m]["格式"]
        L.append(f"| `{m}` | {c.get('SUPPORTED',0)} | {c.get('PARTIAL',0)} | "
                 f"{c.get('CONTRADICTED',0)} |")
    L += ["", "## 三、方向：三件事分列", "",
          "| 模型 | 适用性 | 判定分布 |", "|---|---|---|"]
    for m in ["Base", "L-N3072-3407", "D-N3072-3407"]:
        c = dirs[m]
        app = "；".join(f"{k.split('=')[1]}={v}" for k, v in sorted(c.items())
                       if k.startswith("applicable="))
        ver = "；".join(f"{k.split('=')[1]}={v}" for k, v in sorted(c.items())
                       if k.startswith("verdict="))
        L.append(f"| `{m}` | {app} | {ver} |")
    L += ["", "## 四、边界（不能说的事）", "",
          "1. **这是模型自查，不是专家标注**；本项目已决定不做真人复核，真人栏永久留空。",
          "2. **四维不是独立样本**：56 图 × 4 维来自同样 56 张图，不能当 224 个观测，",
          "   **更不能把某几项的比例说成「描述准确率」**。",
          "3. **格式不是视觉事实**，不与四维加总。",
          "4. 复核者继承本会话上下文（跑过这三个模型、看过本图真值分布），**不冒充独立审判**。",
          "5. **本 56 图没有外部大模型原答**；不与旧 36 图的外部分数直接比较。",
          "6. 与旧 AI 意见（Codex 的 D36 逐图复核等）分歧**原样保留**，未做平均或裁定谁对。"]
    (D / "AI描述验收.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("写出 AI描述验收.json / AI描述验收.md\n")

    print(f"{'模型':<16}{'四维好':>7}{'四维部分':>9}{'四维坏':>7}{'格式好':>7}{'格式坏':>7}")
    for m in ["Base", "L-N3072-3407", "D-N3072-3407"]:
        t = Counter()
        for d in DIMS:
            t.update(per[m][d])
        c = per[m]["格式"]
        print(f"{m:<16}{t.get('SUPPORTED',0):>7}{t.get('PARTIAL',0):>9}"
              f"{t.get('CONTRADICTED',0):>7}{c.get('SUPPORTED',0):>7}{c.get('CONTRADICTED',0):>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
