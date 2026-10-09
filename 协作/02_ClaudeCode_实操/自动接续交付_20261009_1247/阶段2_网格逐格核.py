#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 2：**逐格**核对枚举结果网格（只读主会话目录）。

为什么单独再写一个：阶段 1 的 `枚举复核独立汇总.py` 只比对了「逐模型逐维度」的
**计数**（240 格）。计数一致仍可能掩盖「两个格子彼此换位」这类错误，
所以这里按 **schema 逐格**（每个 模型×图×维度 的取值）直接比。

核对对象（当前时点，不是历史时点）：
  · `复核结果/分片{1..6}_复核.json`  ← 6 片 × 6 图 × 8 模型 = 288 条目
  · `描述对照_枚举.json`            ← 主会话**修复后**成品，`逐条` 1728 = 8×36×6

网格真实形状（**照实记录，不硬凑**）：
  · 枚举网格 = **8 模型 × 36 图 × 6 维 = 1728 格**
  · 类别网格 = **9 类 × 4 图 × 8 模型 = 288 格**（9 类各 4 张，共 36 图）
  ⚠️ 若把「9×36」读成 9 模型 × 36 图，与实际不符 —— 本脚本按实际形状核，
     并在结论里写明真实维度，不把 8 写成 9。

只读，不写主会话目录。
"""
from __future__ import annotations
import hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
MAIN = ROOT / "协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203"

SHARD_DIR = MAIN / "复核结果"
ENUM_JSON = MAIN / "描述对照_枚举.json"
KEY_JSON = MAIN / "盲号对照_key.json"
SCHEMA = MAIN / "复核schema_冻结.json"

DIMS = ["形态", "位置", "遗漏", "断言", "格式", "方向"]
VALUES = {"SUPPORTED", "PARTIAL", "CONTRADICTED", "UNCERTAIN", "MISSING", "NEEDS_REVIEW"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mtime(p: Path) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")


def flat_answers(shard: dict):
    """把一片统一成 [(图号, 盲号, 答案dict), ...]；扁平与分组两种形态都吃。"""
    out = []
    for it in shard.get("条目", []) or []:
        sid = it.get("sample_id") or it.get("图号")
        if isinstance(it.get("答案集"), list):           # 分组格式
            for a in it["答案集"]:
                out.append((sid, a.get("盲号"), a))
        elif "盲号" in it:                                # 扁平格式
            out.append((sid, it.get("盲号"), it))
    for it in shard.get("扁平条目", []) or []:            # 分组格式里的扁平副本
        out.append((it.get("sample_id") or it.get("图号"), it.get("盲号"), it))
    return out


def norm_val(dim: str, v):
    """把一格归一成成品里的字符串写法。

    · 非方向维：dict{verdict, evidence} → verdict
    · 方向维：  dict{applicable, verdict} → 'applicable/verdict'
    """
    if isinstance(v, dict):
        if dim == "方向":
            return f"{str(v.get('applicable')).lower()}/{v.get('verdict')}"
        return v.get("verdict")
    return v


def main() -> int:
    rep: dict = {"目录": str(D), "主会话目录": str(MAIN), "只读": True}

    key_rows = json.loads(KEY_JSON.read_text(encoding="utf-8"))
    blind_map = {r["sample_id"]: r["盲号对照"] for r in key_rows}
    truth = {r["sample_id"]: r["公开类别"] for r in key_rows}

    # ---------- 1. 分片当前状态（哈希 + 计数） ----------
    shards = sorted(SHARD_DIR.glob("分片*_复核.json"))
    shard_meta, cells = [], []
    per_shard_answers = 0
    dropped: list[tuple[str, str, str]] = []      # 同一 (图,盲号) 的重复存放
    seen: set[tuple[str, str]] = set()
    for p in shards:
        sj = json.loads(p.read_text(encoding="utf-8"))
        ans = flat_answers(sj)
        kept = 0
        for sid, blind, a in ans:
            if (sid, blind) in seen:              # 同批记录的第二份存放（如「扁平条目」副本）
                dropped.append((p.name, sid, blind))
                continue
            seen.add((sid, blind))
            kept += 1
            model = (blind_map.get(sid) or {}).get(blind)
            for dim in DIMS:
                cells.append({"sample_id": sid, "盲号": blind, "模型": model,
                              "类别": truth.get(sid), "维度": dim,
                              "判定": norm_val(dim, a.get(dim))})
        shard_meta.append({
            "文件": p.name, "mtime": mtime(p), "sha256": sha(p),
            "条目数": len(sj.get("条目", []) or []),
            "读到答案": len(ans), "去重后计入": kept,
            "被丢弃的重复存放": len(ans) - kept,
            "形态": "分组(答案集)" if (sj.get("条目") and isinstance(sj["条目"][0].get("答案集"), list)) else "扁平",
        })
        per_shard_answers += kept

    rep["一_分片当前状态"] = {
        "分片数": len(shards),
        "每片去重后答案数": [m["去重后计入"] for m in shard_meta],
        "答案总数": per_shard_answers,
        "分片形态分布": dict(Counter(m["形态"] for m in shard_meta)),
        "丢弃的第二份存放": len(dropped),
        "丢弃明细(前10)": [list(x) for x in dropped[:10]],
        "说明": ("分片 2 同时给了「答案集」与「扁平条目」两份**同一批**记录，"
               "只计一次；不这样去重会把 288 读成 336。"),
        "分片明细": shard_meta,
    }

    # ---------- 2. 覆盖唯一性 ----------
    sid_set = {c["sample_id"] for c in cells}
    model_set = {c["模型"] for c in cells}
    pairs = {(c["sample_id"], c["模型"]) for c in cells}
    rep["二_覆盖"] = {
        "唯一 sample_id": len(sid_set),
        "唯一模型": len(model_set),
        "模型": sorted(model_set),
        "模型×图 应有": len(sid_set) * len(model_set),
        "模型×图 实有": len(pairs),
        "未映射盲号": sorted({c["盲号"] for c in cells if not c["模型"]}),
        "覆盖判定": "完整无重复" if (len(pairs) == len(sid_set) * len(model_set)
                                    and not any(not c["模型"] for c in cells)) else "**有问题**",
    }

    # ---------- 3. 枚举网格逐格 ----------
    grid = {(c["sample_id"], c["模型"], c["维度"]): c["判定"] for c in cells}
    expect = {(s, m, d) for s in sid_set for m in model_set for d in DIMS}
    missing_cells = sorted(expect - set(grid))
    illegal = sorted({(k, v) for k, v in grid.items() if v is None})
    bad_dir = sorted({v for (s, m, d), v in grid.items()
                      if d == "方向" and not re.fullmatch(r"(true|false|unknown)/[A-Z_]+", str(v))})
    bad_val = sorted({str(v) for (s, m, d), v in grid.items()
                      if d != "方向" and v not in VALUES})
    rep["三_枚举网格逐格"] = {
        "网格形状": f"{len(model_set)} 模型 × {len(sid_set)} 图 × {len(DIMS)} 维",
        "应有格数": len(expect),
        "实有格数": len(grid),
        "缺格": [list(x) for x in missing_cells[:20]],
        "缺格数": len(missing_cells),
        "空值格数": len(illegal),
        "非枚举值": bad_val,
        "方向格式异常": bad_dir,
        "判定": "逐格齐全且取值合法" if not (missing_cells or illegal or bad_val or bad_dir) else "**有异常**",
    }

    # ---------- 4. 类别网格（9 类 × 4 图） ----------
    cls_img = defaultdict(set)
    for s, c in truth.items():
        if s in sid_set:
            cls_img[c].add(s)
    per_model_cls = defaultdict(Counter)
    for (s, m, d), v in grid.items():
        if d == "形态":
            per_model_cls[m][truth.get(s)] += 1
    rep["四_类别网格"] = {
        "网格形状": f"{len(cls_img)} 类 × {sorted({len(v) for v in cls_img.values()})} 图/类 × {len(model_set)} 模型",
        "类的图数": {k: len(v) for k, v in sorted(cls_img.items())},
        "每模型总数": {m: sum(c.values()) for m, c in sorted(per_model_cls.items())},
        "9类各4图": sorted({len(v) for v in cls_img.values()}) == [4] and len(cls_img) == 9,
        "每模型36": sorted({sum(c.values()) for c in per_model_cls.values()}) == [36],
    }

    # ---------- 5. 与主会话**修复后**成品逐格比 ----------
    cmp_rep: dict = {"对照文件": ENUM_JSON.name,
                     "对照 mtime": mtime(ENUM_JSON), "对照 sha256": sha(ENUM_JSON)}
    if ENUM_JSON.exists():
        ej = json.loads(ENUM_JSON.read_text(encoding="utf-8"))
        rows = ej.get("逐条") or []
        theirs = {(r["sample_id"], r["模型"], r["维度"]): r["判定"] for r in rows}
        diffs = []
        for k in sorted(set(grid) | set(theirs)):
            if grid.get(k) != theirs.get(k):
                diffs.append({"键": list(k), "独立复算": grid.get(k), "成品": theirs.get(k)})
        cmp_rep.update({
            "成品逐条数": len(rows),
            "成品格数(去重)": len(theirs),
            "差异格数": len(diffs),
            "差异明细(前20)": diffs[:20],
            "结论": "逐格一致" if not diffs else "**有差异**",
        })
    rep["五_与成品逐格对比"] = cmp_rep

    ok = (rep["二_覆盖"]["覆盖判定"] == "完整无重复"
          and rep["三_枚举网格逐格"]["判定"].startswith("逐格齐全")
          and rep["四_类别网格"]["9类各4图"] and rep["四_类别网格"]["每模型36"]
          and cmp_rep.get("差异格数") == 0)
    rep["总判定"] = "通过" if ok else "**未通过**"
    rep["时点声明"] = ("以上均为**执行当时**读取的文件状态（分片与成品 mtime/sha256 已随本结果留存）；"
                     "此后主会话若再改动，需重跑本脚本，不能沿用本结论。")

    (D / "阶段2_网格逐格核.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# 逐格核对：枚举结果网格（当前时点）", "",
          f"**总判定：{rep['总判定']}**", "",
          "## 一、分片当前状态（不是历史时点）", "",
          "| 分片 | mtime | 条目 | 读到 | 计入 | 形态 | sha256(前12) |", "|---|---|---:|---:|---|---|"]
    for m in shard_meta:
        md.append(f"| {m['文件']} | {m['mtime']} | {m['条目数']} | {m['读到答案']} | "
                  f"{m['去重后计入']} | {m['形态']} | `{m['sha256'][:12]}` |")
    md += ["",
           f"合计 **{per_shard_answers}** 条答案；形态分布 {rep['一_分片当前状态']['分片形态分布']}", "",
           "## 二、覆盖", "",
           f"- 唯一图 **{rep['二_覆盖']['唯一 sample_id']}** × 唯一模型 **{len(model_set)}** "
           f"→ 应有 {rep['二_覆盖']['模型×图 应有']} 对，实有 {rep['二_覆盖']['模型×图 实有']}",
           f"- 丢弃的第二份存放 **{len(dropped)}**（同一批记录的两份存放，只计一次）；"
           f"未映射盲号 **{len(rep['二_覆盖']['未映射盲号'])}**",
           f"- 判定：**{rep['二_覆盖']['覆盖判定']}**", "",
           "## 三、枚举网格逐格", "",
           f"- 形状：**{rep['三_枚举网格逐格']['网格形状']}**（应有 {len(expect)} 格，"
           f"实有 {len(grid)} 格，缺 {len(missing_cells)}）",
           f"- 非枚举值 {bad_val or '无'}；方向格式异常 {bad_dir or '无'}",
           f"- 判定：**{rep['三_枚举网格逐格']['判定']}**", "",
           "## 四、类别网格", "",
           f"- 形状：{rep['四_类别网格']['网格形状']}；9 类各 4 图 = "
           f"{rep['四_类别网格']['9类各4图']}；每模型 36 图 = {rep['四_类别网格']['每模型36']}",
           "",
           "## 五、与主会话修复后成品逐格对比", "",
           f"- 对照：`{ENUM_JSON.name}`（mtime {cmp_rep.get('对照 mtime')}，"
           f"sha256 `{str(cmp_rep.get('对照 sha256'))[:12]}…`）",
           f"- 成品逐条 **{cmp_rep.get('成品逐条数')}** 条；**差异格数 "
           f"{cmp_rep.get('差异格数')}** → **{cmp_rep.get('结论')}**", "",
           f"> ⚠️ 网格真实形状是 **8 模型 × 36 图**（枚举 1728 格）+ **9 类 × 4 图**（类别 288 格）；",
           "> 「9 模型」不成立 —— 本表按实际形状核，未把 8 写成 9。", "",
           f"> {rep['时点声明']}", ""]
    (D / "阶段2_网格逐格核.md").write_text("\n".join(md), encoding="utf-8")

    sys.stdout.buffer.write((json.dumps({
        "总判定": rep["总判定"],
        "分片数": len(shards), "答案总数": per_shard_answers,
        "覆盖": rep["二_覆盖"]["覆盖判定"],
        "枚举网格": rep["三_枚举网格逐格"]["判定"],
        "网格形状": rep["三_枚举网格逐格"]["网格形状"],
        "与成品差异格数": cmp_rep.get("差异格数"),
        "成品sha256前12": str(cmp_rep.get("对照 sha256"))[:12],
    }, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
