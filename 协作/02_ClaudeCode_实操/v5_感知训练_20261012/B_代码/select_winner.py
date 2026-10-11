#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SFT 赢家选择（B 卡 9；合同 6.2 逐条照做）。
输入：eval/<线>/select200/metrics.json（S0–S4）与 L 的 select200 metrics.json；runs 目录。
输出：winner.json {"winner","checkpoint","config","scores","guard","reason"}。
规则：①六线中 S1p 不参选；②GeoScore 取 metrics.geoscore；③类别守门：该线 SELECT200 类别准确率
≥ L 的类别准确率 − 0.10；④通过守门的线里选 GeoScore 最高，并列取编号小（S0<S1<S2<S3<S4）；
⑤完成 375 步的线才参选（有 checkpoint-375）；都不过/都无 → 选 S1（用其最大已保存 checkpoint）。
结果写 night/winner.json，之后不得再改选。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINES_ORDER = ["S0", "S1", "S2", "S3", "S4"]


def find_max_checkpoint(line_dir: Path):
    best = None
    for p in line_dir.rglob("checkpoint-*"):
        try:
            step = int(p.name.split("checkpoint-")[1])
        except (IndexError, ValueError):
            continue
        if best is None or step > best[0]:
            best = (step, p)
    return best


def choose(candidates, l_acc, guard_gap=0.10):
    """candidates: {line: {"geoscore":..., "class_acc":..., "complete":bool}}
    返回 (winner, guard_info, reason)。"""
    guard = {ln: {"class_acc": c["class_acc"], "l_acc": l_acc,
                  "threshold": (l_acc - guard_gap) if l_acc is not None else None,
                  "passed": ((c["class_acc"] is not None and c["class_acc"] >= l_acc - guard_gap)
                             if l_acc is not None else None)}
             for ln, c in candidates.items()}
    eligible = [ln for ln in LINES_ORDER if ln in candidates and candidates[ln]["complete"]]
    passing = [ln for ln in eligible if guard[ln]["passed"] is not False
               and candidates[ln]["geoscore"] is not None]
    if passing:
        best = max(passing, key=lambda ln: (candidates[ln]["geoscore"], -LINES_ORDER.index(ln)))
        return best, guard, (f"通过守门且 GeoScore 最高（{candidates[best]['geoscore']}）；"
                             f"参选线 {eligible}")
    reason = "无完成 375 步且通过守门的线" if not passing else "候选为空"
    return "S1", guard, f"回退 S1（{reason}）；参选线 {eligible}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--l-metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    l_acc = None
    l_path = Path(a.l_metrics)
    if l_path.exists():
        l_acc = json.loads(l_path.read_text(encoding="utf-8"))["class"]["acc"]
    candidates = {}
    for ln in LINES_ORDER:
        mp = Path(a.eval_root) / ln / "select200" / "metrics.json"
        run_dir = Path(a.runs_root) / ln
        maxck = find_max_checkpoint(run_dir) if run_dir.exists() else None
        complete = bool(maxck and maxck[0] == 375)
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            candidates[ln] = {"geoscore": m.get("geoscore"), "class_acc": m["class"]["acc"],
                              "metrics": str(mp), "complete": complete,
                              "max_step": maxck[0] if maxck else None}
        else:
            candidates[ln] = {"geoscore": None, "class_acc": None, "metrics": None,
                              "complete": complete, "max_step": maxck[0] if maxck else None}
    winner, guard, reason = choose(candidates, l_acc)
    ck = find_max_checkpoint(Path(a.runs_root) / winner)
    out = {"winner": winner, "checkpoint": str(ck[1]) if ck else None,
           "checkpoint_step": ck[0] if ck else None,
           "config": f"configs/{winner}.json", "scores": candidates, "guard": guard,
           "l_acc": l_acc, "reason": reason,
           "rule": "GeoScore 最高者（守门 类别 acc ≥ L−0.10；并列取编号小；都不过选 S1）；不得再改选"}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"赢家 = {winner}（checkpoint {out['checkpoint']}，step {out['checkpoint_step']}）")
    print(reason)
    if ck is None:
        print("**没有可用 checkpoint，夜跑无法继续**")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
