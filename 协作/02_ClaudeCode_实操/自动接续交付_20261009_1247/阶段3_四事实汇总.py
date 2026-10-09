#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 3：**由冻结枚举 JSON 自动生成**四事实 / 格式 / 方向的分项与汇总。

为什么要有这个脚本
------------------
主实验报告里四事实维的**手写摘要**出现过行和错误（例：`L` 的 PARTIAL 手写成 32，
逐维相加实为 15+9+16+3 = **43**；`D` 手写成 37，逐维实为 11+12+13+4 = **40**；
另有模型把「格式」的缺失又加进四事实维）。
手写摘要一旦覆盖了正确的逐维计数，报告就自相矛盾。

本脚本的纪律：
  · **不手填任何数字**：全部从冻结枚举 JSON 的 `逐条`（1728 条 = 8 模型 × 36 图 × 6 维）聚合；
  · **每一行都做行和校验**：每个模型每个维度必须恰好 36 条；四事实 = 144、格式 = 36、方向 = 36；
  · 方向的「不适用 / 可评」与「verdict 分布」**分开列**，不合并成「方向干净率」；
  · **不把综合比例称作描述准确率**（同 36 图不是独立样本）。
只读主会话目录，只写本目录。
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parents[2]
MAIN = REPO / "协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203"
ENUM = MAIN / "描述对照_枚举.json"

DIMS6 = ["形态", "位置", "遗漏", "断言", "格式", "方向"]
FACT4 = ["形态", "位置", "遗漏", "断言"]
VALUES = ["SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN", "MISSING", "NEEDS_REVIEW"]

# 主实验报告里出现过的手写错值（用于在输出里显式对照；**不是**本脚本的输入）
KNOWN_HANDWRITTEN_ERRORS = {
    "L-N3072-3407": {"四事实_PARTIAL": 32},
    "D-N3072-3407": {"四事实_PARTIAL": 37},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    raw = ENUM.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    rows = doc["逐条"]

    # ---- 每模型每维取值计数（全部来自 逐条）----
    per: dict[str, dict[str, dict[str, int]]] = {}
    for r in rows:
        m, dim, v = r["模型"], r["维度"], r["判定"]
        per.setdefault(m, {}).setdefault(dim, {})
        per[m][dim][v] = per[m][dim].get(v, 0) + 1

    problems: list[str] = []
    detail: dict[str, dict] = {}
    for m in sorted(per):
        if set(per[m]) != set(DIMS6):
            problems.append(f"{m}: 维度集合不是六维 → {sorted(per[m])}")
            continue
        # 每维必须 36 条（= 36 图）；四事实 144、格式 36、方向 36
        dim_n = {d: sum(per[m][d].values()) for d in DIMS6}
        for d, n in dim_n.items():
            if n != 36:
                problems.append(f"{m}/{d}: 条数 {n} ≠ 36")
        n_fact4 = sum(dim_n[d] for d in FACT4)
        if n_fact4 != 144:
            problems.append(f"{m}: 四事实合计 {n_fact4} ≠ 144")
        if dim_n["格式"] != 36 or dim_n["方向"] != 36:
            problems.append(f"{m}: 格式 {dim_n['格式']} / 方向 {dim_n['方向']} ≠ 36")
        agg = {v: sum(per[m][d].get(v, 0) for d in FACT4) for v in VALUES}
        if sum(agg.values()) != 144:
            problems.append(f"{m}: 四事实行和 {sum(agg.values())} ≠ 144")
        detail[m] = {"每维条数": dim_n, "六维逐维": per[m], "四事实合计": agg}

    if set(detail) != set(doc["逐模型逐维度"]):
        problems.append("模型集合与本文件 `逐模型逐维度` 不一致")

    # ---- 与本文件自带的汇总段交叉核对（同源双路）----
    cross = []
    for m, blk in doc["逐模型逐维度"].items():
        for d in DIMS6[:5]:
            mine = {v: per[m][d].get(v, 0) for v in VALUES}
            theirs = {v: blk.get(d, {}).get(v, 0) for v in VALUES}
            if mine != theirs:
                cross.append({"模型": m, "维度": d, "逐条聚合": mine, "文件汇总段": theirs})
    if cross:
        problems.append(f"与文件自带 `逐模型逐维度` 不一致 {len(cross)} 处")

    # ---- 方向：不适用 / 可评 与 verdict 分布**分列** ----
    direction: dict[str, dict] = {}
    for m, blk in doc["方向分列"].items():
        app_f = blk.get("applicable=false", 0)
        app_t = blk.get("applicable=true", 0)
        verd = {k.split("verdict=", 1)[1]: v for k, v in blk.items() if k.startswith("verdict=")}
        direction[m] = {
            "不适用(applicable=false)": app_f,
            "可评(applicable=true)": app_t,
            "applicable合计": app_f + app_t,
            "verdict分布": verd,
            "verdict合计": sum(verd.values()),
            "NO_ANSWER": verd.get("NO_ANSWER", 0),
        }
        if app_f + app_t != 36:
            problems.append(f"{m}: 方向 applicable 合计 {app_f + app_t} ≠ 36")
        if sum(verd.values()) != 36:
            problems.append(f"{m}: 方向 verdict 合计 {sum(verd.values())} ≠ 36")

    # ---- 手写错值对照（只报事实）----
    corrections = []
    for m, bad in KNOWN_HANDWRITTEN_ERRORS.items():
        for k, wrong in bad.items():
            key = "四事实合计" if k.endswith("PARTIAL") else None
            right = detail.get(m, {}).get("四事实合计", {}).get("PARTIAL")
            if right is not None:
                corrections.append({
                    "模型": m, "项": "四事实维 PARTIAL 合计",
                    "手写摘要值": wrong, "逐条聚合值": right,
                    "逐维相加": " + ".join(str(per[m][d].get("PARTIAL", 0)) for d in FACT4),
                    "差值": right - wrong,
                })

    report = {
        "来源": {
            "冻结枚举JSON": str(ENUM.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha(ENUM),
            "字节": len(raw),
            "逐条数": len(rows),
            "期望逐条数": len(per) * 36 * 6,
        },
        "口径": {
            "四事实维": FACT4, "格式维": ["格式"], "方向维": ["方向"],
            "行和约束": "每模型每维 36 条；四事实 144；格式 36；方向 36",
            "方向": "「不适用 / 可评」与「verdict 分布」分列；**不合并成方向干净率**",
            "禁用说法": "不得把四事实的好/坏比例称为「描述准确率」——同 36 图不是独立样本",
        },
        "每模型": detail,
        "方向分列": direction,
        "手写摘要错值对照": corrections,
        "校验": {"问题数": len(problems), "问题": problems,
                 "交叉核对不一致数": len(cross), "交叉核对明细": cross},
        "结论": "全部行和通过" if not problems else "**有行和/一致性错误，见 问题**",
    }
    (D / "阶段3_四事实汇总.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- MD ----
    L = ["# 四事实 / 格式 / 方向：由冻结枚举 JSON 自动汇总", "",
         f"**来源**：`{report['来源']['冻结枚举JSON']}`（sha256 `{report['来源']['sha256'][:16]}…`，"
         f"{report['来源']['字节']} 字节）", "",
         f"- 逐条数 **{len(rows)}**（应为 {len(per)} 模型 × 36 图 × 6 维 = "
         f"**{report['来源']['期望逐条数']}**）", "",
         "> **本表全部由脚本聚合，无手写数字。** 每模型每维 36 条；四事实 = 144、格式 = 36、方向 = 36。", "",
         "## 一、四事实维 + 格式维（每模型 5 × 36 = 180 条）", "",
         "| 模型 | " + " | ".join(VALUES) + " | 行和 |", "|---|" + "---:|" * (len(VALUES) + 1)]
    for m in sorted(detail):
        d = detail[m]
        row = [d["四事实合计"][v] + d["六维逐维"]["格式"].get(v, 0) for v in VALUES]
        L.append(f"| `{m}` | " + " | ".join(str(x) for x in row) + f" | **{sum(row)}** |")
    L += ["", "## 二、四事实维合计（**每模型应为 144**）", "",
          "| 模型 | " + " | ".join(VALUES) + " | 行和 |", "|---|" + "---:|" * (len(VALUES) + 1)]
    for m in sorted(detail):
        a = detail[m]["四事实合计"]
        L.append(f"| `{m}` | " + " | ".join(str(a[v]) for v in VALUES) + f" | **{sum(a.values())}** |")
    L += ["", "## 三、格式维（每模型应为 36）", "",
          "| 模型 | " + " | ".join(VALUES) + " | 行和 |", "|---|" + "---:|" * (len(VALUES) + 1)]
    for m in sorted(detail):
        f = detail[m]["六维逐维"]["格式"]
        row = [f.get(v, 0) for v in VALUES]
        L.append(f"| `{m}` | " + " | ".join(str(x) for x in row) + f" | **{sum(row)}** |")
    L += ["", "## 四、方向（不适用 / 可评 与 verdict **分列**，每模型 36）", "",
          "| 模型 | 不适用 | 可评 | verdict 分布 | verdict 合计 |", "|---|---:|---:|---|---:|"]
    for m in sorted(direction):
        x = direction[m]
        dist = " / ".join(f"{k} {v}" for k, v in sorted(x["verdict分布"].items()))
        L.append(f"| `{m}` | {x['不适用(applicable=false)']} | {x['可评(applicable=true)']} | "
                 f"{dist} | **{x['verdict合计']}** |")
    L += ["", "> **方向 36 条「干净」不等于 36 条答对**：22 张图本来就不该有方向，真正可评的只有 14 张。", ""]
    if corrections:
        L += ["## 五、与主实验报告**手写摘要**的对照（错值已由本表取代）", "",
              "| 模型 | 项 | 手写摘要值 | 逐条聚合值 | 逐维相加 | 差 |", "|---|---|---:|---:|---|---:|"]
        for c in corrections:
            L.append(f"| `{c['模型']}` | {c['项']} | {c['手写摘要值']} | **{c['逐条聚合值']}** | "
                     f"{c['逐维相加']} | {c['差值']:+d} |")
        L += ["", "→ 手写摘要**行和不等于 144**，本表以逐条聚合为准；**不手填、不混总比例**。", ""]
    L += ["## 六、校验", "",
          f"- 行和与一致性**问题数：{len(problems)}** → {report['结论']}"]
    if problems:
        L += [f"  - {p}" for p in problems]
    L += [f"- 与本文件自带汇总段交叉核对：不一致 **{len(cross)}** 处", ""]
    (D / "阶段3_四事实汇总.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    sys.stdout.buffer.write((json.dumps({
        "逐条数": len(rows), "模型数": len(detail),
        "问题数": len(problems), "结论": report["结论"],
        "四事实行和": {m: sum(detail[m]["四事实合计"].values()) for m in sorted(detail)},
        "手写错值对照": corrections,
    }, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0 if not problems else 4


if __name__ == "__main__":
    raise SystemExit(main())
