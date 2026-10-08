#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三个复核分片的结果，按**原盲号映射**折回模型并汇总（**v2：修正判据**）。

Codex 复核发现 `汇总复核结果.py` 的判据有错：
`ISSUE` 正则里含「遗漏」「无依据」等**问题词**，于是
**「无重大遗漏。」被判成「有问题」** —— 否定句被当成了肯定句。
（至少影响 D 图13/21/25、L 图14/18。）

本版把判据从「字符串里出现过问题词吗」改成 **从句级 + 否定感知**：

  1. 按 `。；;！？` 切句；
  2. 每句里找**问题词**；若该问题词**前面 6 字内出现否定**
     （无/没有/未/不是/不存在/未见/否/非/毫无/并无），则该处**不算问题**；
  3. 汇总该条的句子：
     · 无任何未被否定的问题词 → **干净**
     · 全是未被否定的问题词 → **有问题**
     · 两者都有 → **混合**（单独一档，不与「有问题」混计）

**混合与不确定分开列，不以命中「遗漏」二字判错。**

⚠️ 另外两个第一版就踩过的坑（保留说明）：
  · **六维极性不同**：「是/否」不能通用一套，按维度分别定义「干净」长什么样；
  · 三个分片 JSON 结构不同（键名「或」vs「/」、单元格 `{结论}` vs `{适用性,吻合性}`），
    本脚本结构无关地解析。

跑法：python 汇总复核结果v2.py
"""
from __future__ import annotations
import io, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "复核结果"
MAP = ROOT / "C_描述事实审阅" / "样本清单.json"
OUT_MD = HERE / "复核汇总v2.md"
OUT_JS = HERE / "复核汇总v2.json"

SIX = ["主要形态是否有图像支持", "位置是否吻合", "方向是否适用/吻合",
       "主要结构是否遗漏", "是否无依据断言", "格式约定问题"]

# ── 判据一：「干净」长什么样（**按维度分别写**，因为极性不同） ──
CLEAN_LEAD = {
    # 「部分…」在**正向维度**里是一个正向信号（部分成立），归入「混合」由
    # 下面的 prob/clean 双真判定处理；在**问题类维度**里「部分遗漏」本身
    # 就含问题词，不会被这里误收。
    "主要形态是否有图像支持": r"^(支持|基本支持|部分支持|符合|吻合|是|✓|有图像支持)",
    "位置是否吻合":          r"^(吻合|基本吻合|部分吻合|一致|符合|是|✓)",
    "方向是否适用/吻合":     r"^(不适用|适用且|适用|吻合|正确|符合|是|✓)",
    "主要结构是否遗漏":      r"^(无|没有|未|否|✗|非)",   # 问「是否遗漏」→ 无 = 干净
    "是否无依据断言":        r"^(无|没有|未|否|✗|非)",   # 问「是否有断言」→ 无 = 干净
    "格式约定问题":          r"^(无|没有|未|否|✗|非)",   # 问「有无问题」→ 无 = 干净
}

# ── 判据二：**问题词**（出现且**未被否定**才算问题） ──
PROBLEM = re.compile(
    r"遗漏|缺失|漏掉|无依据|凭空|与图不符|不符|不吻合|不支持|误报|错误|"
    r"越界|枚举拼接|围栏|不实|相悖|不一致|未填|填错|裸数字|数量词|"
    r"字符串\s*\"?null|过粗|低估|高估|夸大|"
    # 方向维度专有：**在不适用的情况下仍然填了方向**
    r"填了[^。；]*不适用|不适用[^。；]*却填|方向不适用[^。；]*填|"
    r"按约定[^。；]*应为\s*(JSON\s*)?null|环[^。；]*应填\s*null|自述「环形」却填")
NEG = re.compile(r"无|没有|未|不是|不存在|未见|否|非|毫无|并无|不")
# 否定窗口：问题词**前面这么多字符内**出现否定词，就算这一处被否定了
NEG_WIN = 6
CLAUSE = re.compile(r"[^。；;！？\n]+")
UNCERTAIN = re.compile(r"不确定|无法判断|判不了|存疑|无法确定|难以判断")

# ── 回归用例（Codex 点名的五条，加否定/肯定/混合对照） ──
REGRESSION = [
    # (维度, 判定原文, 期望)
    ("主要结构是否遗漏", "无重大遗漏。", "干净"),
    ("主要结构是否遗漏", "无重大结构遗漏（本图无环/线）。", "干净"),
    ("主要结构是否遗漏", "有重大遗漏——漏掉左缘红色块。", "有问题"),
    ("主要结构是否遗漏", "无。", "干净"),
    ("主要结构是否遗漏", "部分遗漏。", "有问题"),
    ("是否无依据断言", "无。", "干净"),
    ("是否无依据断言", "无凭空断言；「稀疏」属与图不符的数量词（已在形态项记录）。", "混合"),
    ("是否无依据断言", "有（轻度）——「零星」属与图不符的数量词。", "有问题"),
    ("格式约定问题", "无。", "干净"),
    ("格式约定问题", '有——字符串 "null"（应为 JSON null）。', "有问题"),
    ("主要形态是否有图像支持", "支持——「随机分布的红色像素点」与高密度散布相符（无环无线）。", "干净"),
    ("主要形态是否有图像支持", "部分支持——「随机点（散布）」成立；但「稀疏」与图不符。", "混合"),
    ("位置是否吻合", "吻合——global 与「全图」相符。", "干净"),
    ("位置是否吻合", "不吻合——实测重心在圆心。", "有问题"),
    ("方向是否适用/吻合", "不适用（无环、无成形长线），填 null 正确。", "干净"),
    ("方向是否适用/吻合", "填了「4点钟」但该图为中心团簇，方向不适用。", "有问题"),
    ("位置是否吻合", "不确定——radial_zone=none 的语义题面未定义。", "不确定"),
]


def clause_has_problem(c: str) -> bool:
    """该句里是否存在**未被否定**的问题词。"""
    for m in PROBLEM.finditer(c):
        pre = c[max(0, m.start() - NEG_WIN):m.start()]
        if NEG.search(pre):
            continue                      # 否定覆盖 → 这一处不算问题
        return True
    return False


def clause_has_clean(c: str, dim: str) -> bool:
    return bool(re.match(CLEAN_LEAD[dim], c.strip()))


def classify(dim: str, v: str) -> str:
    v = (v or "").strip()
    if not v:
        return "缺失"
    if UNCERTAIN.search(v):
        return "不确定"
    cl = [c.strip() for c in CLAUSE.findall(v) if c.strip()]
    prob = any(clause_has_problem(c) for c in cl)
    clean = any(clause_has_clean(c, dim) for c in cl)
    if prob and clean:
        return "混合"
    if prob:
        return "有问题"
    if clean:
        return "干净"
    # 没有肯定信号、也没有未否定的问题词（例如「已填 JSON null，正确」）
    return "干净" if not re.search(r"有(?!效)|存在", v) else "有问题"


# ── 结构无关的解析（三个分片结构不同） ──
def norm_dim(k: str):
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
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if v.get("结论"):
            return str(v["结论"])
        parts = [str(x) for k, x in v.items() if k not in ("证据", "evidence", "说明")]
        return "；".join(parts) if parts else str(v.get("证据", ""))
    return str(v)


def extract(d):
    out = []

    def take(num, blind, src: dict):
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
    print("=" * 74)
    print("一、判据回归（Codex 点名的五条 + 对照）")
    print("=" * 74)
    bad = 0
    for dim, txt, exp in REGRESSION:
        got = classify(dim, txt)
        flag = "✓" if got == exp else "✗"
        if got != exp:
            bad += 1
        print(f"  {flag} [{dim[:6]}] {txt[:34]:<36} 期望={exp:<5} 得到={got}")
    print(f"  回归：{len(REGRESSION)-bad}/{len(REGRESSION)} 通过\n")
    if bad:
        print("  **回归未过，不继续汇总**")
        return 2

    mapping = json.loads(io.open(MAP, encoding="utf-8").read())
    key = {s["图号"]: s["盲号对照"] for s in mapping["样本"]}

    rows = []
    for i in (1, 2, 3):
        p = RES / f"分片{i}_复核.json"
        if not p.exists():
            print(f"  !! 缺 {p.name}"); continue
        got = extract(json.loads(io.open(p, encoding="utf-8").read()))
        print(f"  分片{i}: {len(got)} 个答案集")
        rows += got
    print(f"  合计 {len(rows)} 个答案集（应为 108）\n")

    per = defaultdict(lambda: defaultdict(Counter))
    detail = []
    for num, blind, f in rows:
        m = (key.get(num) or {}).get(blind)
        if m is None:
            print(f"  !! 图 {num} 盲号 {blind} 映射缺失"); continue
        for dim in SIX:
            txt = f.get(dim, "")
            lab = classify(dim, txt)
            per[m][dim][lab] += 1
            detail.append({"图号": num, "盲号": blind, "模型": m, "维度": dim,
                           "判定": lab, "原文": txt[:240]})

    io.open(OUT_JS, "w", encoding="utf-8", newline="\n").write(
        json.dumps({"判据": "从句级 + 否定感知；混合单列；见脚本头注释",
                    "逐模型逐维度": {m: {d: dict(c) for d, c in v.items()}
                                 for m, v in per.items()},
                    "逐条": detail}, ensure_ascii=False, indent=2) + "\n")

    L = ["# 36 图描述事实复核汇总 v2（按盲号映射折回模型）", "",
         "> **全部是模型复核草稿，不是人工 gold。真人两列仍然空白。**",
         "> **这是 v2**：v1 的判据把「无重大遗漏」当成「有问题」（命中了问题词但没看否定），",
         "> 本版改成**从句级 + 否定感知**，并把**混合**单独列一档。",
         "> 原始判定文字与逐图证据见 `复核结果/分片{1,2,3}_复核.md`。", "",
         "## 口径", "",
         "1. **六维极性不同**：",
         "   「主要形态是否有图像支持 / 位置是否吻合 / 方向是否适用吻合」→"
         " **支持/吻合才是干净**；",
         "   「主要结构是否遗漏 / 是否无依据断言 / 格式约定问题」→ **「无」才是干净**。",
         "2. **判据**：按 `。；;！？` 切句，逐句找问题词；"
         "**问题词前 6 字内有否定词的，该处不算问题**。",
         "3. **混合** = 同一条里既有干净句、又有未否定的问题句。**与「有问题」分开计**，"
         "不合并、也不当作判错。",
         "4. **不确定** = 复核者自己说判不了（允许且鼓励；本次仅 7 条）。",
         "5. **本表不评价类别对错**；材料里没有类别，复核时也没给。",
         "6. 每模型的**可评项数 = 6 维度 × 12 图 × 3 答案集 = 216**；"
         "三模型合计 **648** 项。**不是 648 张独立样本**。", "",
         "## 一、按模型（可评项数 = 216/模型；648 项为汇总数）", ""]
    TOT = Counter()
    for m in sorted(per):
        L += [f"### `{m}`", "",
              "| 维度 | 干净 | 混合 | 有问题 | 不确定 |", "|---|---:|---:|---:|---:|"]
        for dim in SIX:
            c = per[m].get(dim, {})
            for k, v in c.items():
                TOT[k] += v
            L.append(f"| {dim} | {c.get('干净',0)} | {c.get('混合',0)} | "
                     f"{c.get('有问题',0)} | {c.get('不确定',0)} |")
        s = Counter()
        for c in per[m].values():
            s.update(c)
        L += [f"| **小计** | **{s['干净']}** | **{s['混合']}** | "
              f"**{s['有问题']}** | **{s['不确定']}** |", ""]
    L += ["## 二、三模型合计", "",
          "| | 干净 | 混合 | 有问题 | 不确定 | 合计 |", "|---|---:|---:|---:|---:|---:|",
          f"| 648 项 | {TOT['干净']} | {TOT['混合']} | {TOT['有问题']} | "
          f"{TOT['不确定']} | **{sum(TOT.values())}** |", "",
          "## 三、v1 → v2 的变化（只改判据，**未重做 36 图复核**）", ""]
    io.open(OUT_MD, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")

    print(f"写出 {OUT_MD.name} / {OUT_JS.name}")
    print(f"\n{'模型':<18}{'干净':>6}{'混合':>6}{'有问题':>8}{'不确定':>8}")
    for m in sorted(per):
        s = Counter()
        for c in per[m].values():
            s.update(c)
        print(f"{m:<18}{s['干净']:>6}{s['混合']:>6}{s['有问题']:>8}{s['不确定']:>8}")
    print(f"{'合计':<18}{TOT['干净']:>6}{TOT['混合']:>6}{TOT['有问题']:>8}{TOT['不确定']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
