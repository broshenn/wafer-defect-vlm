#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多字段 RL v2 首轮：从确认池原答重算指标（CPU，可复跑）。

输入：`out_v2/评测/{M0,M1,M2}_raw.jsonl` + `out_v2/confirm120_gold.jsonl`
输出：`out_v2/首轮指标.json` 与 `out_v2/首轮指标.md`

口径（与运行合同一致）：
  · 每个字段单独算，另报格式通过率；
  · **方向只在该字段对参考适用（eligible）的样本上算**，并写明适用样本数；
  · 同图配对差用 bootstrap（seed 3407）给区间；
  · **不把各字段合成一个"总准确率"** —— 自由文本 caption 不参与任何分数。
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os as _os
# 自测用：WAFER_V2_DIR 直接指向含 {MODEL}_raw.jsonl 与 confirm120_gold.jsonl 的目录
V2DIR = Path(_os.environ["WAFER_V2_DIR"]) if _os.environ.get("WAFER_V2_DIR") else HERE / "out_v2"
sys.path.insert(0, str(HERE))
from 奖励_v2 import parse_answer, score_one, parse_reference, check_format  # noqa: E402

MODELS = ["M0", "M1", "M2"]
FIELDS = ["class", "coverage_level", "zones", "direction"]


def boot_ci(diffs, n=10000, seed=3407):
    rng = random.Random(seed)
    k = len(diffs)
    means = []
    for _ in range(n):
        s = sum(diffs[rng.randrange(k)] for _ in range(k)) / k
        means.append(s)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)], sum(diffs) / k


def main() -> int:
    gold = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (V2DIR / "confirm120_gold.jsonl").open(encoding="utf-8")}
    per_model = {}
    raw_ok = {}
    for m in MODELS:
        p = V2DIR / "评测" / f"{m}_raw.jsonl"
        if not p.exists():
            print(f"缺少 {p} —— 未运行或未下载"); continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        got = {}
        for r in rows:
            sid = r["sample_id"]
            if sid not in gold:
                continue
            ref = parse_reference({k: gold[sid][k] for k in
                                   ("gold_class", "gold_coverage_level", "gold_zones",
                                    "gold_direction_type", "gold_sectors")}, 0)
            obj, how = parse_answer(r["raw"])
            res = score_one(obj, ref)
            fmt, _ = check_format(obj, how)
            got[sid] = {"parts": res["parts"], "elig": ref["elig"],
                        "fact": res["fact"], "fmt": fmt, "parse": how,
                        "reward": 0.9 * res["fact"] + 0.1 * fmt}
        per_model[m] = got
        raw_ok[m] = len(got)

    if not per_model:
        print("没有任何原答，退出"); return 0

    n_common = min(len(v) for v in per_model.values())
    common = sorted(set.intersection(*[set(v) for v in per_model.values()]))[:n_common]

    out = {"确认池样本数": len(gold), "各模型原答数": raw_ok,
           "三模型共同样本数": len(common), "逐模型": {}, "配对差": {}}

    for m, got in per_model.items():
        acc = {}
        for f in FIELDS:
            elig = [s for s in common if got[s]["elig"][f]]
            hit = [s for s in elig if got[s]["parts"][f] == 1.0]
            # 方向分是 0.6/0.4 加权，类型对不全等于 1.0；单列类型准确率
            typed = [s for s in elig if got[s]["parts"][f] is not None]
            acc[f] = {"适用样本": len(elig), "完全正确": len(hit),
                      "准确率": round(len(hit) / len(elig), 4) if elig else None}
        # 方向类型单独看
        dt_elig = [s for s in common if got[s]["elig"]["direction"]]
        dt_ok = [s for s in dt_elig if got[s]["parts"]["direction"] >= 0.6]
        acc["direction_type_only"] = {"适用样本": len(dt_elig), "类型正确": len(dt_ok),
                                      "准确率": round(len(dt_ok) / len(dt_elig), 4)
                                      if dt_elig else None}
        acc["格式"] = {"平均格式分": round(sum(got[s]["fmt"] for s in common) / len(common), 4),
                       "解析失败": sum(1 for s in common if got[s]["parse"] != "ok")}
        acc["奖励"] = {"平均奖励": round(sum(got[s]["reward"] for s in common) / len(common), 4)}
        out["逐模型"][m] = acc

    for pair in (("M1", "M0"), ("M2", "M0"), ("M2", "M1")):
        a, b = pair
        if a not in per_model or b not in per_model:
            continue
        d = [per_model[a][s]["reward"] - per_model[b][s]["reward"] for s in common]
        lo, hi, mean = boot_ci(d)
        # 类别字段的配对差
        dc = [per_model[a][s]["parts"]["class"] - per_model[b][s]["parts"]["class"]
              for s in common]
        lo2, hi2, mean2 = boot_ci(dc)
        out["配对差"][f"{a}-{b}"] = {
            "奖励差均值": round(mean, 4), "奖励差95%区间": [round(lo, 4), round(hi, 4)],
            "区间跨0": bool(lo <= 0 <= hi),
            "类别字段差均值": round(mean2, 4),
            "类别字段差95%区间": [round(lo2, 4), round(hi2, 4)],
            "类别字段差区间跨0": bool(lo2 <= 0 <= hi2)}

    (V2DIR / "首轮指标.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 多字段 RL v2 首轮指标（确认池 120 图）", "",
             f"三模型共同样本 {len(common)} / 确认池 {len(gold)}", "",
             "| 模型 | 类别 | 覆盖档位 | 径向区带 | 方向(加权) | 方向类型 | 格式均分 | 平均奖励 |",
             "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        if m not in out["逐模型"]:
            continue
        a = out["逐模型"][m]
        def f(x):
            v = a[x]["准确率"]
            return f"{v:.4f}（{a[x]['完全正确']}/{a[x]['适用样本']}）" if v is not None else "n/a"
        dt = a["direction_type_only"]
        lines.append(f"| {m} | {f('class')} | {f('coverage_level')} | {f('zones')} | "
                     f"{f('direction')} | "
                     f"{dt['准确率']:.4f}（{dt['类型正确']}/{dt['适用样本']}） | "
                     f"{a['格式']['平均格式分']:.4f} | {a['奖励']['平均奖励']:.4f} |")
    lines += ["", "## 配对差（bootstrap 10000，seed 3407）", "",
              "| 对比 | 奖励差均值 | 95% 区间 | 跨 0？ | 类别字段差 | 95% 区间 | 跨 0？ |",
              "|---|---:|---|---|---:|---|---|"]
    for k, v in out["配对差"].items():
        lines.append(f"| {k} | {v['奖励差均值']:+.4f} | "
                     f"[{v['奖励差95%区间'][0]:+.4f}, {v['奖励差95%区间'][1]:+.4f}] | "
                     f"{'是' if v['区间跨0'] else '否'} | {v['类别字段差均值']:+.4f} | "
                     f"[{v['类别字段差95%区间'][0]:+.4f}, {v['类别字段差95%区间'][1]:+.4f}] | "
                     f"{'是' if v['类别字段差区间跨0'] else '否'} |")
    (V2DIR / "首轮指标.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
