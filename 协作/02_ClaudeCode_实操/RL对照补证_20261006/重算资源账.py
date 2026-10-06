#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重算资源账：**按每张卡的占用区间并集**，不把同一张卡同一段墙钟算两次。
同时把「进程分钟」与「物理设备分钟」分列。
"""
from __future__ import annotations
import io, json, datetime
from pathlib import Path

DST = Path("D:/pycode/圆图研究")  # placeholder, 下面覆盖
DST = Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/RL对照补证_20261006")


def t(hhmmss):
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return datetime.timedelta(hours=h, minutes=m, seconds=s)


def un(intervals):
    """区间并集总长（分钟）。"""
    iv = sorted(intervals)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += (ce - cs).total_seconds() / 60
            cs, ce = s, e
    if cs is not None:
        tot += (ce - cs).total_seconds() / 60
    return round(tot, 2)


# ── 区间定义 ────────────────────────────────────────────
# 10-02 旧基座推理：device_map=cuda:0 → 只占 GPU 0
OLD = [("旧基座推理 10-02（单卡 GPU0）", [("0", t("03:16:55"), t("03:25:50"))])]

# 第一轮（六卡 device_map auto）+ 占位进程
HOLD_A = ("holder", t("01:07:24"), t("01:23:15"))     # GPU 0 与 1
HOLD_B = ("holder", t("01:23:15"), t("01:41:10"))     # 仅 GPU 1
RUNS = [("smoke 首次失败", t("01:23:34"), t("01:24:37")),
        ("smoke 成功", t("01:27:00"), t("01:28:28")),
        ("主训练 60 步", t("01:29:32"), t("01:36:47")),
        ("Adapter dev18 推理", t("01:38:24"), t("01:39:01"))]

per_card = {str(i): [] for i in range(6)}
per_card["0"].append((HOLD_A[1], HOLD_A[2]))
per_card["1"].append((HOLD_A[1], HOLD_A[2]))
per_card["1"].append((HOLD_B[1], HOLD_B[2]))
for _, s, e in RUNS:
    for c in "012345":
        per_card[c].append((s, e))

R1 = {c: un(v) for c, v in per_card.items()}
r1_total = round(sum(R1.values()), 2)

# 第二轮（五个单卡任务，互不重叠；卡 0/1/2/4/5 各一段）
R2 = [("V 验证+信息移除", "0", 3.79), ("M 合并等价", "1", 0.98),
      ("A1 训练", "2", 5.97), ("A2 训练", "4", 6.20), ("A3 训练", "0", 5.95),
      ("A1/A2/A3 推理", "1/2/4", 6.55)]
r2_total = round(sum(x[2] for x in R2), 2)

# 第三轮（RL，单卡串行）
R3 = [("E 阶段采样", 1.12), ("F2 两步小测试", 0.77), ("F2 20 步正式", 3.07),
      ("F1 20 步对照", 2.03), ("统一评测 F0", 1.60), ("统一评测 F1/F2", 3.52),
      ("失败尝试 ×2（有日志）", 0.38)]
r3_total = round(sum(x[1] for x in R3), 2)

old_total = round(sum(un([(s, e)]) for _, iv in OLD for _, s, e in iv), 2)
RESERVE = 1.0   # 一次更早的失败尝试无独立日志，保守预留

total = round(old_total + r1_total + r2_total + r3_total + RESERVE, 2)

out = {
    "上限": 180,
    "口径": {
        "物理设备分钟": "按**每张卡的占用区间并集**计算；同一张卡同一段墙钟只算一次。",
        "进程分钟": "各进程自身墙钟之和；多卡进程按进程计一次。两者**不混称**。",
    },
    "0_旧基座推理_10-02": {"设备分钟": old_total, "设备": "GPU0（device_map=cuda:0）"},
    "1_第一轮_六卡": {
        "逐卡并集分钟": R1,
        "设备分钟": r1_total,
        "去重说明": "占位进程（01:07:24–01:41:10，GPU0+1）与六卡训练/推理窗口（01:23:34–01:39:01）"
                    "在同一物理卡上**时间重叠**：GPU1 的并集被占位区间完全覆盖；"
                    "GPU0 的占位段与训练段不相交，故两段相加。**未重复累计。**",
        "朴素相加（错误口径，仅作对照）": round(71.26 + 49.62, 2),
    },
    "2_第二轮_五任务单卡": {"明细": [{"项": a, "卡": b, "设备分钟": c} for a, b, c in R2],
                            "设备分钟": r2_total},
    "3_第三轮_RL": {"明细": [{"项": a, "设备分钟": b} for a, b in R3], "设备分钟": r3_total},
    "保守预留_无日志失败": RESERVE,
    "累计设备分钟": total,
    "剩余": round(180 - total, 2),
    "本轮新增上限": 8,
    "备注": "旧报告 162.81 用的是**朴素相加**（六卡 71.26 + 占位 49.62），"
            "与按卡并集的口径相差约 19 分钟；本文件为重算后的口径。",
}
DST.mkdir(parents=True, exist_ok=True)
(DST / "资源账_进程与设备分钟.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

print("逐卡并集（第一轮）:", R1, "小计", r1_total)
print("10-02:", old_total, "| 第二轮:", r2_total, "| 第三轮:", r3_total, "| 预留:", RESERVE)
print(f"累计 {total} / 180   剩余 {round(180-total,2)}")
print("（朴素相加口径会是", round(71.26 + 49.62 + old_total + r2_total + r3_total + RESERVE, 2), "）")
