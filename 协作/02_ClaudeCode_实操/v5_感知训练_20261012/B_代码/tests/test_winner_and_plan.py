#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""select_winner 规则测试（正常、并列、都不过守门、缺失线）+ make_night_plan 计划检验。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B))
from select_winner import choose  # noqa: E402


def c(geo, acc, complete=True, max_step=375):
    return {"geoscore": geo, "class_acc": acc, "complete": complete, "max_step": max_step}


def main():
    results = []
    add = lambda n, cond, x="": results.append((n, bool(cond), x))  # noqa: E731

    cands = {"S0": c(0.50, 0.60), "S1": c(0.55, 0.65), "S2": c(0.53, 0.62),
             "S3": c(0.40, 0.70), "S4": c(0.45, 0.55)}
    w, g, why = choose(cands, l_acc=0.62)
    add("正常：GeoScore 最高者胜（S1）", w == "S1", (w, why))

    cands2 = {"S0": c(0.55, 0.60), "S1": c(0.55, 0.65), "S2": c(0.53, 0.62),
              "S3": c(0.40, 0.70), "S4": c(0.45, 0.55)}
    w, _, _ = choose(cands2, l_acc=0.62)
    add("并列取编号小（S0）", w == "S0", w)

    cands3 = {"S0": c(0.50, 0.30), "S1": c(0.55, 0.40), "S2": c(0.53, 0.35),
              "S3": c(0.40, 0.30), "S4": c(0.45, 0.20)}
    w, g, why = choose(cands3, l_acc=0.62)
    add("都不过守门 → S1", w == "S1" and "回退" in why, (w, why))

    cands4 = {"S0": None, "S1": c(0.55, 0.65), "S2": None, "S3": None, "S4": None}
    cands4 = {"S0": {"geoscore": None, "class_acc": None, "complete": True, "max_step": 375},
              "S1": c(0.55, 0.65),
              "S2": {"geoscore": None, "class_acc": None, "complete": False, "max_step": 250},
              "S3": {"geoscore": None, "class_acc": None, "complete": False, "max_step": 100},
              "S4": {"geoscore": None, "class_acc": None, "complete": False, "max_step": 125}}
    w, _, _ = choose(cands4, l_acc=0.62)
    add("缺失线不参选（S1 胜）", w == "S1", w)

    cands5 = {"S0": c(0.90, 0.80, complete=False, max_step=250), "S1": c(0.20, 0.65),
              "S2": None, "S3": None, "S4": None}
    cands5 = {"S0": {"geoscore": 0.90, "class_acc": 0.80, "complete": False, "max_step": 250},
              "S1": c(0.20, 0.65), "S2": {"geoscore": None, "class_acc": None, "complete": False, "max_step": None},
              "S3": {"geoscore": None, "class_acc": None, "complete": False, "max_step": None},
              "S4": {"geoscore": None, "class_acc": None, "complete": False, "max_step": None}}
    w, _, _ = choose(cands5, l_acc=0.62)
    add("未完成 375 步不参选（S1 胜）", w == "S1", w)

    # make_night_plan 计划检验
    plan_path = B / "night_plan_test.json"
    scripts_dir = B / "scripts"
    gpus = {f"g{i}": f"uuid-{i}" for i in range(1, 8)}
    (B / "gpus_test.json").write_text(json.dumps(gpus), encoding="utf-8")
    (B / "smoke_timing_test.json").write_text(json.dumps(
        {"sft_s_per_step": 20, "rl_s_per_step": 60, "eval_s_per_img": 6, "load_s": 120,
         "eval_batch_size": 8}), encoding="utf-8")
    import subprocess
    import os
    env = dict(os.environ, WS_ROOT="/WS")
    r = subprocess.run([sys.executable, "-I", str(B / "make_night_plan.py"),
                        "--smoke-timing", str(B / "smoke_timing_test.json"),
                        "--gpus", str(B / "gpus_test.json"),
                        "--out", str(plan_path), "--scripts-dir", str(scripts_dir)],
                       capture_output=True, text=True, env=env)
    add("make_night_plan 退出 0", r.returncode == 0, r.stderr[-300:])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    ids = [j["id"] for j in plan["jobs"]]
    add("任务 id 唯一", len(ids) == len(set(ids)))
    # 无环 + 依赖存在
    depmap = {j["id"]: j["after"] for j in plan["jobs"]}
    seen, stack = set(), set()

    def dfs(n):
        if n in stack:
            return False
        if n in seen:
            return True
        stack.add(n)
        for d in depmap[n]:
            if d not in depmap or not dfs(d):
                return False
        stack.discard(n)
        seen.add(n)
        return True

    add("无环且依赖全部存在", all(dfs(n) for n in depmap))
    gpu_jobs = [j for j in plan["jobs"] if j["gpus"] != ["cpu"]]
    add("每个 GPU 任务都有卡", all(j["gpus"] and all(g in gpus for g in j["gpus"]) for j in gpu_jobs))
    by = {j["id"]: j for j in plan["jobs"]}
    scr = Path(by["eval_S3_test200"]["cmd"][1]).read_text(encoding="utf-8")
    scr_sel = Path(by["eval_S3_select"]["cmd"][1]).read_text(encoding="utf-8")
    add("S3 评测用 _x2 数据", "--dataset /WS/v5/data/test200_main_x2.jsonl" in scr
        and "--dataset /WS/v5/data/select200_main_x2.jsonl" in scr_sel, scr[:160])
    add("RL 全部依赖 screen_merge",
        all("screen_merge" in by[f"rl_{l}"]["after"] for l in ("RSFT", "Rfull", "Rbinary", "Rnoproc", "Rfullp")))
    add("hard_stop 08:30", plan["hard_stop"] == "08:30")
    add("训练任务标记 training", all(by[i]["training"] for i in ids if i.startswith(("sft_", "rl_", "gate_"))))

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, extra in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name} {extra if not ok else ''}")
    print(f"{n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
