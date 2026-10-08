#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三个复核分片的结果，按**原盲号映射**折回模型并汇总。

Codex 要求：「复核后按原盲号映射汇总到模型，分别报告**可评覆盖 / 不确定项 /
错误类型**；选例用预定 36 图，不能只挑成功。」

**这一步是"事后才允许"的**：复核期间子agent 看不到映射，汇总时才用。

⚠️ **两个必须写清楚的坑**（本脚本第一版两个都踩了）：

1. **六个维度的极性是混的。** 「主要形态是否有图像支持」的「是」是好的，
   而「主要结构**是否遗漏**」「**是否**无依据断言」「格式约定**问题**」的「是」
   是坏的。把六列塞进同一个「是/否」表会**把结论说反**。
   所以本脚本**按维度分别定义**「干净 / 有问题」的判据。

2. **判定文字长短不一**，有「无。」也有整句话。只按首字打标签会把
   `clock_direction 为字符串 "null"。` 这种**明确违规**漏进「其他」。
   所以本脚本**同时输出逐维度的判定原文字频次表**，让分类可被审计。

两个分片用了不同的 JSON 结构，本脚本两种都吃：
  · 结构甲：`条目[].答案集[].六项{维度:{结论,证据}}`
  · 结构乙：`复核条目[]` 每行一个答案集，六个维度是**平铺的字符串**

跑法：python 汇总复核结果.py
"""
from __future__ import annotations
import io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "复核结果"
MAP = ROOT / "C_描述事实审阅" / "样本清单.json"

SIX = ["主要形态是否有图像支持", "位置是否吻合", "方向是否适用/吻合",
       "主要结构是否遗漏", "是否无依据断言", "格式约定问题"]

# 每个维度的「干净」判据。**按维度分别写**，因为极性不同。
CLEAN = {
    "主要形态是否有图像支持": r"^(支持|基本支持|是|✓|吻合|符合|有图像支持)",
    "位置是否吻合":          r"^(吻合|基本吻合|一致|是|✓|符合)",
    "方向是否适用/吻合":     r"^(不适用|适用且|适用|吻合|正确|是|✓|符合)",
    "主要结构是否遗漏":      r"^(无|没有|未|否|✗)",          # 问的是「是否遗漏」→ 无 = 干净
    "是否无依据断言":        r"^(无|没有|未|否|✗)",          # 问的是「是否有断言」→ 无 = 干净
    "格式约定问题":          r"^(无|没有|未|否|✗)",          # 问的是「有无问题」→ 无 = 干净
}
# 明确的「有问题」信号（任意位置出现即算）
ISSUE = re.compile(r"字符串\s*\"?null|裸数字|越界|枚举拼接|不吻合|不支持|与图不符|"
                   r"误报|遗漏|未填|填错|不一致|相悖|不实|失败标注|无依据|"
                   r"应填\s*JSON|null\s*而非|轻度|轻微|部分")
UNCERTAIN = re.compile(r"不确定|无法判断|判不了|存疑")


def classify(dim: str, v: str) -> str:
    v = (v or "").strip()
    if not v:
        return "缺失"
    if UNCERTAIN.search(v):
        return "不确定"
    if re.match(CLEAN[dim], v) and not ISSUE.search(v):
        return "干净"
    return "有问题"


# 三个分片各自用了**不同的键名与嵌套**，这里统一归一化。
#   分片1：条目[].答案集[].六项{「方向是否适用**或**吻合」:{结论,证据}}
#   分片2：条目[].答案集[]{六项**平铺**，「方向是否适用/吻合」}
#   分片3：复核条目[]{六项平铺}
# 只按「方向」二字匹配，避免被「或」/「/」的写法差异坑到 —— 这正是第一版漏掉
# 大量方向判定的原因。
def norm_dim(k: str) -> str | None:
    for dim in SIX:
        if k == dim:
            return dim
    if "方向" in k and "适用" in k:
        return "方向是否适用/吻合"
    if "形态" in k:
        return "主要形态是否有图像支持"
    if "位置" in k and "吻合" in k:
        return "位置是否吻合"
    if "遗漏" in k:
        return "主要结构是否遗漏"
    if "无依据断言" in k:
        return "是否无依据断言"
    if "格式" in k:
        return "格式约定问题"
    return None


def _cell(v):
    """把单元格取值成一段判定文字。三个分片的单元格形状不一样：
       · 字符串（分片2/3）
       · {结论, 证据}（分片1 的多数维度）
       · {适用性, 吻合性, 证据}（分片1 的**方向**维度 —— 第一版漏了这个，
         导致 36 条方向判定全被当成空）
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if v.get("结论"):
            return str(v["结论"])
        parts = [str(x) for k, x in v.items() if k not in ("证据", "evidence", "说明")]
        if parts:
            return "；".join(parts)
        return str(v.get("证据", ""))
    return str(v)


def extract(d):
    """返回 [(图号, 盲号, {维度: 判定原文})]，三种结构都吃。"""
    out = []

    def take(num, blind, src: dict):
        """src 是含六项的字典（可能在 `六项` 里，也可能直接平铺）。"""
        f = {dim: "" for dim in SIX}
        for k, v in src.items():
            dim = norm_dim(k)
            if dim and not f[dim]:
                f[dim] = _cell(v)
        if any(f.values()):
            out.append((num, blind, f))

    for it in d.get("条目", []):
        for a in it.get("答案集", []):
            take(it.get("图号"), a.get("盲号"), a.get("六项") or a)
    for it in d.get("复核条目", []):
        take(it.get("图号"), it.get("盲号"), it)
    return out


def main() -> int:
    mapping = json.loads(io.open(MAP, encoding="utf-8").read())
    key = {s["图号"]: s["盲号对照"] for s in mapping["样本"]}

    rows, missing = [], []
    for i in (1, 2, 3):
        p = RES / f"分片{i}_复核.json"
        if not p.exists():
            missing.append(p.name); continue
        got = extract(json.loads(io.open(p, encoding="utf-8").read()))
        print(f"  分片{i}: {len(got)} 个答案集")
        rows += got
    if missing:
        print(f"  !! 尚缺 {missing}")
    print(f"  合计 {len(rows)} 个答案集（应为 108）")

    per = defaultdict(lambda: defaultdict(Counter))
    verbatim = defaultdict(Counter)          # (模型,维度) -> 判定原文
    detail = []
    for num, blind, f in rows:
        m = (key.get(num) or {}).get(blind)
        if m is None:
            print(f"  !! 图 {num} 盲号 {blind} 映射缺失"); continue
        for dim in SIX:
            txt = f.get(dim, "")
            lab = classify(dim, txt)
            per[m][dim][lab] += 1
            verbatim[(m, dim)][txt.strip()[:60]] += 1
            detail.append({"图号": num, "盲号": blind, "模型": m, "维度": dim,
                           "判定": lab, "原文": txt[:240]})

    io.open(HERE / "复核汇总.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"逐模型逐维度": {m: {d: dict(c) for d, c in v.items()}
                                 for m, v in per.items()},
                    "逐条": detail,
                    "口径": {
                        "极性": "六维极性不同；「是否遗漏/是否断言/格式问题」的"
                                "『无』才是干净。见脚本 CLEAN 表。",
                        "不确定": "允许且鼓励；判不出来必须写它",
                        "类别": "本表不评价类别对错"}},
                   ensure_ascii=False, indent=2) + "\n")

    L = ["# 36 图描述事实复核汇总（按盲号映射折回模型）", "",
         "> **全部是模型复核草稿，不是人工 gold。真人两列仍然空白。**",
         "> 复核期间三个子agent 互不读取映射；本表是汇总阶段才折回去的。",
         "> 六项判定的原始文字与逐图证据见 `复核结果/分片{1,2,3}_复核.md`。", "",
         "## ⚠️ 读表之前必须先知道：**六个维度的极性是混的**", "",
         "| 维度 | 「干净」长什么样 |",
         "|---|---|",
         "| 主要形态是否有图像支持 | 支持 / 基本支持 |",
         "| 位置是否吻合 | 吻合 / 基本吻合 |",
         "| 方向是否适用、吻合 | 不适用（方向本就不适用时）/ 适用且吻合 |",
         "| **主要结构是否遗漏** | **「无遗漏」才是干净** |",
         "| **是否无依据断言** | **「无断言」才是干净** |",
         "| **格式约定问题** | **「无问题」才是干净** |",
         "",
         "把六列塞进同一个「是/否」表会**把结论说反**，所以上表按维度分别定义。", "",
         "## 一、按模型汇总（216 项 = 3 模型 × 72 项；每项 = 6 维度 × 12 图）", ""]
    for m in sorted(per):
        L += [f"### `{m}`", "", "| 维度 | 干净 | 有问题 | 不确定 |", "|---|---:|---:|---:|"]
        for dim in SIX:
            c = per[m].get(dim, {})
            L.append(f"| {dim} | {c.get('干净',0)} | {c.get('有问题',0)} | "
                     f"{c.get('不确定',0)} |")
        L.append("")
    L += ["## 二、判定原文字频次（**分类可被审计**；同一条可能同时含干净与问题）", ""]
    for m in sorted(per):
        L += [f"### `{m}`", ""]
        for dim in SIX:
            top = verbatim[(m, dim)].most_common(4)
            L.append(f"- **{dim}**：" +
                     "；".join(f"「{k}」×{v}" for k, v in top if k) or "（无）")
        L.append("")
    L += ["## 三、口径", "",
          "1. **「不确定」是允许且鼓励的答案** —— 没有可靠依据就必须写它，**不算错**。",
          "2. **本表不评价类别对错**；材料里没有类别，复核时也没给。",
          "3. 环形 / 全片 / 中心团簇 / 无缺陷的图，方向一栏应为「不适用」。",
          "4. **选例时不能只挑成功案例** —— 逐图证据与问题清单见各分片 `.md`。",
          "5. 三个分片各用了一套内部结构（`条目/答案集/六项` 与 `复核条目` 平铺），"
          "本脚本两种都解析；**判定文字保留原文，未改写**。",
          "6. 本汇总由 `汇总复核结果.py` 生成，可重跑核对。"]
    io.open(HERE / "复核汇总.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")

    print(f"\n写出 复核汇总.md / 复核汇总.json")
    print(f"\n{'模型':<18}{'干净':>6}{'有问题':>8}{'不确定':>8}")
    for m in sorted(per):
        a = sum(c.get("干净", 0) for c in per[m].values())
        b = sum(c.get("有问题", 0) for c in per[m].values())
        u = sum(c.get("不确定", 0) for c in per[m].values())
        print(f"{m:<18}{a:>6}{b:>8}{u:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
