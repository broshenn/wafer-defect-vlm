#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从两份结果 JSON 生成 模型结果总表.md（避免手抄数字）。"""
from __future__ import annotations
import json
from pathlib import Path

D = Path(__file__).resolve().parent
cat = json.loads((D / "类别与格式结果.json").read_text(encoding="utf-8"))
ai = json.loads((D / "AI描述验收.json").read_text(encoding="utf-8"))
TAGS = ["Base", "L-N3072-3407", "D-N3072-3407"]
DIMS = ["形态", "位置", "遗漏", "断言"]
rows = []
for t in TAGS:
    c = cat["模型"][t]; f = c["格式"]; a = ai["逐模型逐维度"][t]
    four = {k: sum(a[d].get(k, 0) for d in DIMS)
            for k in ["SUPPORTED", "PARTIAL", "CONTRADICTED"]}
    fm = {k: a["格式"].get(k, 0) for k in ["SUPPORTED", "PARTIAL", "CONTRADICTED"]}
    ds = ai["方向分列"][t]
    rows.append({
        "t": t, "N": c["N"],
        "acc": c["类别"]["严格Acc"], "mf1": c["类别"]["严格MacroF1_七类"],
        "facc": c["类别"]["围栏Acc"], "ci": c["类别"]["Acc_bootstrap95"],
        "mf1_9": c["类别"]["九类口径"]["MacroF1"],
        "schema_ok": f["schema全合法"], "warn": sum(f["约定警告计数"].values()),
        "f4": four, "ff": fm,
        "dapp": ds.get("applicable=true", 0), "dsup": ds.get("verdict=SUPPORTED", 0),
        "dna": ds.get("verdict=NO_ANSWER", 0)})

L = []
A = L.append
A("# 模型结果总表（56 图隔离候选，2026-10-09）")
A("")
A("**性质**：七类、可见历史隔离后的**留出候选补测**，**不是独立九类最终考卷**"
  "（缺 Near_full / none，且完整历史接触未核）。")
A("真值 = 公开 manifest 的 failure_type（label_source 为 ground_truth），只用于离线评分，**未下发给模型**。")
A("")
A("## 一、类别（严格 JSON 口径；未解析按类别错误计）")
A("")
A("| 模型 | N | 严格 Acc | Macro-F1（七类） | Acc 95% bootstrap | 围栏 Acc | 九类口径 Macro-F1 |")
A("|---|---:|---:|---:|---|---:|---:|")
for r in rows:
    A(f"| `{r['t']}` | {r['N']} | {r['acc']:.4f} | {r['mf1']:.4f} | "
      f"[{r['ci'][0]}, {r['ci'][1]}] | {r['facc']:.4f} | {r['mf1_9']:.4f} |")
A("")
A("> 九类口径把缺席的 Near_full / none 按 F1=0 计入分母，**仅是本 56 图七类覆盖下的一个口径**，")
A("> 不构成九类全面覆盖，也不与旧 36 / 84 图成绩换分母比较。")
A("")
A("## 二、格式（与视觉事实**分列**）")
A("")
A("| 模型 | 严格 JSON | schema 全合法 | 约定警告（clock 写成字符串 null） |")
A("|---|---:|---:|---:|")
for r in rows:
    A(f"| `{r['t']}` | {r['N']}/{r['N']} | {r['schema_ok']}/{r['N']} | {r['warn']} |")
A("")
A("## 三、AI 描述验收（**模型自查，非人工 gold**；56 图 × 4 维 = 224 项）")
A("")
A("| 模型 | 四维 好 | 部分 | 坏 | 格式 好 | 格式 坏 |")
A("|---|---:|---:|---:|---:|---:|")
for r in rows:
    A(f"| `{r['t']}` | {r['f4']['SUPPORTED']} | {r['f4']['PARTIAL']} | "
      f"{r['f4']['CONTRADICTED']} | {r['ff']['SUPPORTED']} | {r['ff']['CONTRADICTED']} |")
A("")
A("> 四维来自同样 56 张图，**不是 224 个独立观测**；不得把比例说成「描述准确率」。")
A("> 复核者为执行会话本人，亲自跑过这三个模型 —— **不是严格盲评**。")
A("")
A("## 四、方向（三件事分列）")
A("")
A("| 模型 | 可评（applicable=true） | 判对 | 未给方向 |")
A("|---|---:|---:|---:|")
for r in rows:
    A(f"| `{r['t']}` | {r['dapp']} | {r['dsup']} | {r['dna']} |")
A("")
A("> 「方向干净」不等于「方向答对」：不适用 / 未答 / 可评且判对三个数必须分列。")
A("")
A("## 五、同图配对差（bootstrap 10000 次，seed 3407）")
A("")
A("| 对比 | 均值差 | 95% 区间 | 同对同错 |")
A("|---|---:|---|---:|")
for k, v in cat["同图配对差"].items():
    A(f"| {k} | {v['均值差(严格Acc)']:+.4f} | {v['bootstrap95']} | {v['同对同错']} |")
A("")
A("> **L − D 的区间跨 0**：在本 56 图上**分不出 L 与 D 的类别高下**。")
A("> L − Base 与 D − Base 的区间**不含 0**；同基座、同题面、同解码的监督增益在本轮可复现。")
A("")
(D / "模型结果总表.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("写出 模型结果总表.md")
