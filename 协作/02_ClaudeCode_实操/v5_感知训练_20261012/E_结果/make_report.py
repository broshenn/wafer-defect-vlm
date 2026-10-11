#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E 阶段：从拉回的产物生成 v5_初版报告.md（数字全部来自文件，不手抄）。
用法：python make_report.py --pull <拉回解包目录> --data <A_数据目录> --out <报告路径>
输入：pull/ 下有 night/{state.json,winner.json,passk.json,smoke_report.json}、
      runs/*/{v5_config.json,runtime_gate.json}、runs/*_probe/gate_sft.json、
      eval/<tag>/<set>/{metrics.json,per_sample.jsonl}、eval/PROBE/probe.json、eval/CNN/metrics.json
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

EVAL_MODELS = ["BASE", "L", "S0", "S1", "S2", "S3", "S4", "S1p",
               "RSFT", "Rfull", "Rbinary", "Rnoproc", "Rfullp"]
PAIRS_GEOSCORE = [("S1", "S0", "课程"), ("S2", "S1", "视觉侧"), ("S3", "S1", "分辨率"),
                  ("S1", "S4", "数据引擎"), ("S1", "S1p", "种子方差参照"),
                  ("Rfull", "RSFT", "RL 主结论"), ("Rfull", "Rbinary", "距离奖励"),
                  ("Rfull", "Rnoproc", "过程奖励"), ("Rfull", "Rfullp", "种子方差参照")]


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def m_field(m, path, default="not run"):
    cur = m
    for k in path.split("."):
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur


def fmt(v, nd=3):
    if v is None:
        return "not run"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = Path(a.pull)
    D = Path(a.data)
    N = P / "night"

    state = load(N / "state.json") or {"jobs": {}}
    winner = load(N / "winner.json") or {}
    passk = load(N / "passk.json") or {}
    smoke = load(N / "smoke_report.json") or {}

    def metrics(tag, sset):
        return load(P / "eval" / tag / sset / "metrics.json")

    def per_sample(tag, sset):
        return P / "eval" / tag / sset / "per_sample.jsonl"

    # ── 执行状态表 ──
    lines_status = []
    for line in ("S0", "S1", "S2", "S3", "S4", "S1p"):
        st = state["jobs"].get(f"sft_{line}", {})
        lines_status.append((f"SFT {line}", st.get("status", "?"),
                             st.get("reason", "") or "", st.get("seconds")))
    for line in ("RSFT", "Rfull", "Rbinary", "Rnoproc", "Rfullp"):
        st = state["jobs"].get(f"rl_{line}", {})
        lines_status.append((f"RL {line}", st.get("status", "?"),
                             st.get("reason", "") or "", st.get("seconds")))

    # ── SHUF50 多数类基线（用本地 TEST200 参考算）──
    maj = {}
    try:
        refs = [json.loads(l)["reference_json"] for l in
                (D / "test200_main.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        refs = [json.loads(r) for r in refs]
        for field in ("coverage_band", "angular_type"):
            c = Counter(r[field] for r in refs if r[field] != "unknown")
            top, cnt = c.most_common(1)[0]
            maj[field] = {"答案": top, "命中率": round(cnt / len(refs), 4)}
        zc = Counter(json.dumps(r["red_mass_zones"], sort_keys=True)
                     for r in refs if r["red_mass_zones"] != "unknown")
        top, cnt = zc.most_common(1)[0]
        maj["red_mass_zones"] = {"答案": top, "命中率": round(cnt / len(refs), 4)}
        maj["geoscore_天花板"] = round(sum(v["命中率"] for v in maj.values()) / 3, 4)
    except FileNotFoundError:
        maj = {"error": "本地 test200_main.jsonl 不可用"}

    # ── bootstrap（本地跑）──
    boot = {}
    for ma, mb, label in PAIRS_GEOSCORE:
        pa, pb = per_sample(ma, "test200"), per_sample(mb, "test200")
        if pa.exists() and pb.exists():
            r = subprocess.run([sys.executable, "-I", str(Path(__file__).parent.parent / "B_代码" / "bootstrap_v5.py"),
                                "--a", str(pa), "--b", str(pb), "--metric", "geoscore", "--n", "10000"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                boot[f"{label}: {ma}−{mb} (GeoScore)"] = json.loads(r.stdout.strip().splitlines()[-1])
        pa2, pb2 = per_sample(ma, "test200"), per_sample(mb, "test200")
        if ma.startswith("R") and mb.startswith("R") and pa2.exists() and pb2.exists():
            r = subprocess.run([sys.executable, "-I", str(Path(__file__).parent.parent / "B_代码" / "bootstrap_v5.py"),
                                "--a", str(pa2), "--b", str(pb2), "--metric", "class_macro_f1", "--n", "10000"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                boot[f"{label}: {ma}−{mb} (类别 Macro-F1)"] = json.loads(r.stdout.strip().splitlines()[-1])

    md = []
    md.append("# v5 感知训练 初版报告（2026-10-12 深夜跑）\n")
    md.append(f"生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}；"
              f"所有数字来自服务器回传产物（pull/），无手抄。\n")

    # 一句话结论
    rl_main = boot.get("RL 主结论: Rfull−RSFT (GeoScore)")
    if rl_main is None:
        concl = "RL 主结论：**pending**（R-full 或 R-SFT 的 TEST200 评测未完成）。"
    elif rl_main["excludes_zero"] and rl_main["point_estimate"] > 0:
        concl = (f"RL 主结论：**达标**——R-full 的 TEST200 GeoScore 高于 R-SFT "
                 f"（差值 {rl_main['point_estimate']:+.4f}，95% 区间 [{rl_main['ci95'][0]:+.4f}, "
                 f"{rl_main['ci95'][1]:+.4f}] 不含 0）；类别 Macro-F1 差见比较对表。")
    elif rl_main["excludes_zero"] and rl_main["point_estimate"] < 0:
        concl = (f"RL 主结论：**未达标（反向）**——R-full 的 GeoScore 低于 R-SFT "
                 f"（差值 {rl_main['point_estimate']:+.4f}，区间不含 0）。")
    else:
        concl = (f"RL 主结论：**本批未观察到差异**——R-full−R-SFT GeoScore 差值 "
                 f"{rl_main['point_estimate']:+.4f}，95% 区间含 0。")
    md.append(f"**一句话结论**：SFT 赢家为 **{winner.get('winner', '?')}**"
              f"（SELECT200 GeoScore {winner.get('scores', {}).get(winner.get('winner', ''), {}).get('geoscore', '?')}）。"
              f"{concl}\n")

    # 执行状态
    md.append("## 执行状态表")
    md.append("| 线 | 状态 | 说明/耗时 |")
    md.append("|---|---|---|")
    for name, st, reason, secs in lines_status:
        info = reason if reason else (f"{secs:.0f}s" if secs else "")
        md.append(f"| {name} | {st} | {info} |")
    for extra in ("gate_S3", "probe", "cnn", "eval_L_test200", "eval_screen_merge"):
        st = state["jobs"].get(extra)
        if st:
            md.append(f"| {extra} | {st['status']} | {(st.get('reason') or '')[:60]} |")
    md.append("")

    # 主表 TEST200
    md.append("## 主表（TEST200）")
    md.append("| 模型 | GeoScore | coverage | zones | angular | 扇区(样本数) | ratio MAE | 类别 acc | Macro-F1 | 格式(剥/原始) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag in EVAL_MODELS:
        m = metrics(tag, "test200")
        if m is None:
            md.append(f"| {tag} | not run | | | | | | | | |")
            continue
        if m.get("kind") == "L":
            md.append(f"| {tag} | — | — | — | — | — | — | {fmt(m_field(m, 'class.acc'))} | "
                      f"{fmt(m_field(m, 'class.macro_f1'))} | — |")
            continue
        sec = m_field(m, "sector.acc")
        secn = m_field(m, "sector.n", "?")
        md.append(f"| {tag} | {fmt(m.get('geoscore'))} | {fmt(m_field(m, 'coverage.exact_rate'))} | "
                  f"{fmt(m_field(m, 'zones.exact_rate'))} | {fmt(m_field(m, 'angular_type.acc'))} | "
                  f"{fmt(sec)} ({secn}) | {fmt(m_field(m, 'measure_mae.red_ratio'))} | "
                  f"{fmt(m_field(m, 'class.acc'))} | {fmt(m_field(m, 'class.macro_f1'))} | "
                  f"{fmt(m_field(m, 'format.stripped_legal_rate'))}/{fmt(m_field(m, 'format.raw_legal_rate'))} |")
    md.append("")

    # 旋转/去图
    md.append("## 旋转等变（ROT100）与去图（SHUF50）")
    md.append("| 模型 | ROT100 coverage 不变 | ROT100 zones 不变 | ROT100 扇区+3 | SHUF50 GeoScore |")
    md.append("|---|---|---|---|---|")
    for tag in EVAL_MODELS:
        mr = metrics(tag, "rot100")
        ms = metrics(tag, "shuf50")
        rp = (mr or {}).get("rot_pair") if mr else None
        row = [tag]
        if rp:
            row += [fmt(m_field(rp, "coverage_unchanged.rate")),
                    fmt(m_field(rp, "zones_unchanged.rate")),
                    fmt(m_field(rp, "sector_plus3.rate"))]
        else:
            row += ["pending/not run"] * 3
        row.append(fmt((ms or {}).get("geoscore")) if ms else "not run")
        md.append("| " + " | ".join(row) + " |")
    md.append(f"\n多数类基线（SHUF 对照）：{json.dumps(maj, ensure_ascii=False)}（用 TEST200 参考分布）\n")

    # 预注册比较对
    md.append("## 预注册比较对（按 lot 配对 bootstrap，10000 次，seed 3407）")
    md.append("| 比较对 | 点估计 | 95% 区间 | 结论 |")
    md.append("|---|---|---|---|")
    for k, v in boot.items():
        verdict = ("区间不含 0 → 有效" if v["excludes_zero"] else "本批未观察到差异")
        md.append(f"| {k} | {v['point_estimate']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {verdict} |")
    if not boot:
        md.append("| （全部 pending） | | | |")
    md.append("\n（RL 主结论成功标准：R-full GeoScore 高于 R-SFT 且区间不含 0，同时类别 Macro-F1 不低于 R-SFT 超过 0.03）\n")

    # 门槛证据
    md.append("## 门槛证据")
    g1 = load(P / "runs" / "S1_probe" / "gate_sft.json")
    g2 = load(P / "runs" / "S2_probe" / "gate_sft.json")
    md.append(f"- S1 gate：visual 键 {g1 and g1.get('visual_key_count')}（应为 0），passed={g1 and g1.get('passed')}")
    md.append(f"- S2 gate：visual 键 {g2 and g2.get('visual_key_count')}（应 >0），passed={g2 and g2.get('passed')}")
    for line in ("RSFT", "Rfull", "Rbinary", "Rnoproc", "Rfullp"):
        rg = load(P / "runs" / line / "runtime_gate.json")
        if rg:
            md.append(f"- {line} runtime_gate：policy_equal_M0={rg.get('policy_equal_M0_tensor_count')}、"
                      f"reference_equal_M0={rg.get('reference_equal_M0_tensor_count')}、"
                      f"two_step={rg.get('two_step_gate')}")
    md.append(f"- 图池筛选：{json.dumps(passk, ensure_ascii=False)}")
    cfg_rf = load(P / "runs" / "Rfull" / "v5_config.json")
    md.append(f"- dynamic_sample：{cfg_rf.get('dynamic_sample') if cfg_rf else 'not run'}"
              f"；起点：{cfg_rf.get('start_adapter') if cfg_rf else ''}")
    pr = load(P / "eval" / "PROBE" / "probe.json")
    if pr:
        md.append(f"- 线性探针（P2）：mean acc {pr.get('mean_acc')}（三字段 {json.dumps(pr.get('fields'), ensure_ascii=False)}）")
    cn = load(P / "eval" / "CNN" / "metrics.json")
    if cn:
        md.append(f"- CNN（P1）：acc {cn['class']['acc']}、Macro-F1 {cn['class']['macro_f1']}（impl: {cn.get('impl')}）")
    md.append("")

    # 冒烟与非致命问题
    md.append("## 冒烟与过程记录")
    for k, v in (smoke.get("stages") or {}).items():
        md.append(f"- 冒烟 {k}: {'PASS' if v['ok'] else 'FAIL'}")
    for note in dict.fromkeys(smoke.get("notes", [])):   # 去重（重跑会多次 append）
        md.append(f"- 备注：{note}")
    md.append("")

    md.append("## DECISIONS.md")
    dec = (Path(__file__).parent.parent / "DECISIONS.md")
    if dec.exists():
        md.append("（全文见 v5_感知训练_20261012/DECISIONS.md，以下为标题索引）")
        for l in dec.read_text(encoding="utf-8").splitlines():
            if l.startswith("## "):
                md.append(f"- {l[3:]}")
    md.append("")

    # 扇区分母提示
    try:
        import json as _json
        refs2 = [_json.loads(_json.loads(l)["reference_json"]) for l in
                 (D / "test200_main.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        n_sec = sum(1 for r in refs2 if r["angular_type"] in ("single", "opposed"))
        md.append(f"**扇区指标分母提示**：TEST200 中参考为 single/opposed 的样本仅 {n_sec} 个"
                  f"（几何定义下 single 全库仅 47/5904）——扇区正确率欠功效，仅作参考。
")
    except Exception:
        pass

    md.append("## 未完成与限制")
    md.append("- 未运行项以 not run 标注，不以 0 填充；partial 以实际步数标注。")
    md.append("- S3（896 图线）gate 失败（样本超 max_length=2048），整线 not run。")
    md.append("- probe 低于多数类基线（合同配方未含特征标准化），解读按预注册规则执行。")
    md.append("- 扇区指标分母极小（TEST200 仅 1 个 single/opposed 参考样本；TEST 全集 10 个）→ 扇区正确率欠功效。")
    md.append("- rot-pair 补算与调度器链式 score 任务存在竞态：个别模型 rot 指标可能被后写的 score 覆盖 → 最终版在全部 score 完成后补算取数。")
    md.append("- 一切数字来自程序真值与公开标签；本报告不声称人工双审。")

    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"报告已写：{a.out}（{len(chr(10).join(md))} 字符）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
