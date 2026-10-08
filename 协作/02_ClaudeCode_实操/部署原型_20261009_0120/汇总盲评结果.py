#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署原型 vs 外部 GLM：36 图描述事实复核的汇总（**盲评结果的解锁汇总**）。

材料已盲（只有图号/图片/盲号 + 四个描述字段，无来源、无类别）。
**盲号 → 来源**的映射只在本脚本里用，复核者看不到。

判据沿用本轮已修正的 **从句级 + 否定感知**（`汇总复核结果v2.py`）：
  · 按 `。；;！？` 切句；
  · 问题词**前 6 字内有否定词的，该处不算问题**（修掉「无重大遗漏被判成有问题」）；
  · **混合**单独一档，不与「有问题」混计。
**六维极性不同**：「形态支持 / 位置吻合 / 方向适用吻合」→ 支持才是干净；
「是否遗漏 / 是否断言 / 格式问题」→ **「无」才是干净**。

跑法：python 汇总盲评结果.py
"""
from __future__ import annotations
import io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
RES = D / "复核结果（已盲）"
KEY = D / "盲号对照_key.json"
OUT_MD = D / "描述自查与对照.md"
OUT_JS = D / "描述自查与对照.json"

SIX = ["主要形态是否有图像支持", "位置是否吻合", "方向是否适用/吻合",
       "主要结构是否遗漏", "是否无依据断言", "格式约定问题"]
CLEAN_LEAD = {
    "主要形态是否有图像支持": r"^(支持|基本支持|部分支持|符合|吻合|是|✓|有图像支持)",
    "位置是否吻合":          r"^(吻合|基本吻合|部分吻合|一致|符合|是|✓)",
    "方向是否适用/吻合":     r"^(不适用|适用且|适用|吻合|正确|符合|是|✓)",
    "主要结构是否遗漏":      r"^(无|没有|未|否|✗|非)",
    "是否无依据断言":        r"^(无|没有|未|否|✗|非)",
    "格式约定问题":          r"^(无|没有|未|否|✗|非)",
}
PROBLEM = re.compile(
    r"遗漏|缺失|漏掉|无依据|凭空|与图不符|不符|不吻合|不支持|误报|错误|"
    r"越界|枚举拼接|围栏|不实|相悖|不一致|未填|填错|裸数字|数量词|"
    r"字符串\s*\"?null|过粗|低估|高估|夸大|误识|否认|自相矛盾|"
    r"填了[^。；]*不适用|不适用[^。；]*却填|按约定[^。；]*应为\s*(JSON\s*)?null")
NEG = re.compile(r"无|没有|未|不是|不存在|未见|否|非|毫无|并无|不")
NEG_WIN = 6
CLAUSE = re.compile(r"[^。；;！？\n]+")
UNCERTAIN = re.compile(r"不确定|无法判断|判不了|存疑|无法确定|难以判断")


def clause_problem(c):
    for m in PROBLEM.finditer(c):
        if NEG.search(c[max(0, m.start() - NEG_WIN):m.start()]):
            continue
        return True
    return False


def classify(dim, v):
    v = (v or "").strip()
    if not v:
        return "缺失"
    if UNCERTAIN.search(v):
        return "不确定"
    cl = [c.strip() for c in CLAUSE.findall(v) if c.strip()]
    prob = any(clause_problem(c) for c in cl)
    clean = any(re.match(CLEAN_LEAD[dim], c) for c in cl)
    if prob and clean:
        return "混合"
    if prob:
        return "有问题"
    return "干净"


def cell(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if v.get("结论"):
            return str(v["结论"])
        parts = [str(x) for k, x in v.items() if k not in ("证据", "evidence")]
        return "；".join(parts) if parts else str(v.get("证据", ""))
    return "" if v is None else str(v)


def norm_dim(k):
    for d in SIX:
        if k == d:
            return d
    if "方向" in k and "适用" in k:
        return "方向是否适用/吻合"
    if "形态" in k:
        return "主要形态是否有图像支持"
    if "位置" in k and "吻合" in k:
        return "位置是否吻合"
    if "遗漏" in k:
        return "主要结构是否遗漏"
    if "无依据" in k or "断言" in k:
        return "是否无依据断言"
    if "格式" in k:
        return "格式约定问题"
    return None


def extract(d):
    """三种结构都吃：
       · 条目[].答案集[].六项{维度:...}
       · 条目[] 平铺维度键（本轮盲评就是这个）
       · 复核条目[] 平铺
    """
    out = []

    def take(num, blind, src):
        f = {dim: "" for dim in SIX}
        for k, v in src.items():
            dim = norm_dim(k)
            if dim and not f[dim]:
                f[dim] = cell(v)
        if any(f.values()):
            out.append((num, blind, f))

    for it in d.get("条目", []):
        if it.get("答案集"):
            for a in it["答案集"]:
                take(it.get("图号"), a.get("盲号"), a.get("六项") or a)
        else:
            take(it.get("图号"), it.get("盲号"), it)
    for it in d.get("复核条目", []):
        take(it.get("图号"), it.get("盲号"), it)
    return out


def main():
    key = json.loads(KEY.read_text(encoding="utf-8"))      # 结构: {sample_id: {盲号: 来源}}
    # 图号 → sample_id（key 文件是按 sample_id 存的）
    smp = json.loads((D / "样本清单.json").read_text(encoding="utf-8"))["样本"]
    num2sid = {s["图号"]: s["sample_id"] for s in smp}
    rows, missing = [], []
    for i in (1, 2, 3):
        p = RES / f"分片{i}_复核.json"
        if not p.exists():
            missing.append(p.name); continue
        got = extract(json.loads(p.read_text(encoding="utf-8")))
        print(f"  分片{i}: {len(got)} 个答案集")
        rows += got
    if missing:
        print(f"  !! 尚缺 {missing}"); return 4
    print(f"  合计 {len(rows)} 个答案集（应为 72 = 36 图 × 2）")

    per = defaultdict(lambda: defaultdict(Counter))
    detail = []
    for num, blind, f in rows:
        sid = num2sid.get(num) or num2sid.get(str(num))
        src = (key.get(sid) or {}).get(blind)
        if src is None:
            print(f"  !! 图 {num} 盲号 {blind} 映射缺失"); continue
        for dim in SIX:
            lab = classify(dim, f.get(dim, ""))
            per[src][dim][lab] += 1
            detail.append({"图号": num, "盲号": blind, "来源": src,
                           "维度": dim, "判定": lab, "原文": f.get(dim, "")[:240]})

    OUT_JS.write_text(json.dumps(
        {"口径": {"判据": "从句级 + 否定感知；混合单列", "极性": "六维极性不同，见脚本 CLEAN_LEAD",
                  "盲评": "复核者看不到来源与类别；来源在汇总时才折回",
                  "性质": "模型复核草稿，不是人工 gold；真人复核本项目已决定不做"},
         "逐来源逐维度": {s: {d: dict(c) for d, c in v.items()} for s, v in per.items()},
         "逐条": detail}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L = ["# 描述自查与对照（部署原型 vs 外部 GLM）", "",
         "> **全部是模型复核草稿，不是人工 gold。**",
         "> **本项目已决定不做真人复核（2026-10-08），真人两列永久留空。**", "",
         "## 口径", "",
         "1. **盲评**：复核者只看到图片 + 两个匿名答案集（甲/乙），",
         "   **看不到来源、看不到公开类别**；来源映射只在汇总阶段折回。",
         "2. **判据**：从句级 + 否定感知（问题词前 6 字内有否定则不算问题），"
         "**混合单独一档**。这套判据是 Codex 指出「无重大遗漏被判成有问题」后修正的。",
         "3. **六维极性不同**：「形态支持 / 位置吻合 / 方向适用吻合」→ 支持才是干净；",
         "   「是否遗漏 / 是否断言 / 格式问题」→ **「无」才是干净**。",
         "4. **不判类别对错**；公开类别只用于类别评分，不作为描述 gold。", "",
         "## 一、按来源（各 36 图 × 6 维 = 216 项）", ""]
    for s in sorted(per):
        L += [f"### {s}", "", "| 维度 | 干净 | 混合 | 有问题 | 不确定 |", "|---|---:|---:|---:|---:|"]
        for dim in SIX:
            c = per[s].get(dim, {})
            L.append(f"| {dim} | {c.get('干净',0)} | {c.get('混合',0)} | "
                     f"{c.get('有问题',0)} | {c.get('不确定',0)} |")
        t = Counter()
        for c in per[s].values():
            t.update(c)
        L += [f"| **小计** | **{t['干净']}** | **{t['混合']}** | **{t['有问题']}** | **{t['不确定']}** |", ""]
    L += ["## 二、必须一起写的边界", "",
          "1. **这是模型自查**，执行者是模型且**继承编排会话的上下文**，不能替代人工复核。",
          "2. **36 图是开发集**（val84 每类 2 张 + dev18，`Near_full` 仅 4），**不是新盲测卷**。",
          "3. **外部 GLM 的真实后端标识与生成参数未知**；只写「客户端显示 GLM-5.3-Flash」。",
          "4. **服务功能通过 ≠ 描述事实通过** —— 两者分列。",
          "5. 本表只判**描述与图是否相符**，**不含**类别正确率（那在部署报告里另列）。"]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n写出 {OUT_MD.name} / {OUT_JS.name}")
    print(f"{'来源':<14}{'干净':>6}{'混合':>6}{'有问题':>8}{'不确定':>8}")
    for s in sorted(per):
        t = Counter()
        for c in per[s].values():
            t.update(c)
        print(f"{s:<14}{t['干净']:>6}{t['混合']:>6}{t['有问题']:>8}{t['不确定']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
