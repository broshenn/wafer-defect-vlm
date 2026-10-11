# -*- coding: utf-8 -*-
"""追加记录 · 程序F：生成 汇总与一览 的终版副本（v2 = v1 原样字段 + amendment_v2 声明修正块）。

原 final_summary.json / 30图复核结果一览.md 原件不改动；v2 写入本目录 终版副本/。
21/7/2 模型裁决与逐条分歧原样保留；不称正确率、不称专家审核。
"""
import json
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).parent
ROUND = D.parent
OUT = D / "终版副本"
OUT.mkdir(exist_ok=True)

v1 = json.loads((ROUND / "final_summary.json").read_text(encoding="utf-8"))

amendment = {
    "version": "v2_amended_copy",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "rule": "本文件为 v1(final_summary.json) 的完整副本＋amendment_v2 块；v1 数值字段（含 21/7/2 与 per_item 裁决）未改动。原件保留于上一级目录。",
    "declarations_fixed": [
        {
            "field": "counts.actually_viewed",
            "v1_basis": "按五件套文件存在计数（不充分）",
            "v2_basis": "客户端审计日志：30/30 图各恰 1 次 PNG Read 工具事件（且 started 写入先于 PNG Read）。仍只到工具调用层：后端是否将图像送入模型上下文不可证。",
        },
        {
            "field": "mechanical_validation.obs_before_review_mtime（观察先于旧答打开）",
            "v1_basis": "以 observation mtime ≤ review mtime 证明「打开顺序」（不充分：mtime 只证写盘顺序，不证打开事件）",
            "v2_basis": "客户端审计日志工具事件序：Write(observation_raw) 早于 Read(旧raw) 30/30，且旧 raw 每图恰 1 次 Read。此为客户端工具调用证据；「写入的观察文本=模型当时的首次认知」仍属执行者声明，由原答不改约定与SHA清单约束。",
        },
        {
            "field": "observation_meta.visual_reads=1",
            "v1_basis": "worker 自报",
            "v2_basis": "审计日志每图 PNG Read 工具调用恰 1 次、无 Bash 图像内容查看命令（仅 hashlib 字节读取）。后端侧看图次数不可证。",
        },
        {
            "field": "自报时间（started_at_utc / done_at_utc）",
            "v1_basis": "未单独核",
            "v2_basis": "16 图自报时间链矛盾核实成立（根因：手写估计、本地钟面误标Z）；追加发现片4整片自报时间同为本地误标Z。全部 30 图的执行时刻应以审计日志事件时刻为准；自报时间不得再作执行时刻引用。",
        },
    ],
    "semantic_hold": [
        {
            "item_id": "wbv3_064", "review_id": "rv30_027", "model_verdict": "usable（保留）",
            "issue": "旧答与本轮观察的位置描述均指向右侧绿色合格区（location_clock=[3] 指绿色区）",
            "handling": "描述对象需分开处理：不得在下游把该图位置当作红色缺陷主位置使用；两份模型原答不改；不提供替换性猜测答案。与 Codex 分账 semantic_hold_in_usable 一致。",
        }
    ],
    "hash_correction": {
        "item_id": "wbv3_056",
        "v1_issue": "追加更正说明中『缺少中段 92 两字符，且总长差 3 个字符』表述不准确",
        "v2_basis": "程序difflib结果：单处插入、长度3（规范对齐 0基第27位插 '279'）；无替换、无尾部截断。见 追加记录/程序C_056哈希字符差异.json。只证当前PNG与盲清单一致，不证当时读取。",
    },
    "boundaries_unchanged": [
        "21 usable / 7 requires_review / 2 reject 为模型裁决分布，不是正确率，不是专家审核；不得推广到未复核的其余 54 图。",
        "真实后端型号与采样参数仍未提供；不宣称后端独立已证、专家双审或独立盲评已成立。",
        "本轮为记录补充，不构成内容重新审核。",
    ],
    "artifacts": {
        "audit_timeline": "程序A_审计日志时间线.json",
        "time_contradictions": "程序B_矛盾分类.json + 时间顺序矛盾核查.md",
        "real_log_index": "真实日志索引.md",
        "hash_diff": "程序C_056哈希字符差异.json",
        "evidence_and_downgrade": "证据来源与声明降级.md",
        "wbv3_064_note": "wbv3_064_描述对象说明.md",
    },
}

v2 = dict(v1)
v2["amendment_v2"] = amendment
(OUT / "final_summary_v2.json").write_text(json.dumps(v2, ensure_ascii=False, indent=1), encoding="utf-8")

# 一览 v2：复用 v1 表格内容 + 追加段
v1_md = (ROUND / "30图复核结果一览.md").read_text(encoding="utf-8")
footer = """

---

## v2 追加说明（2026-10-11，只补记录不改裁决）

本表 21/7/2 与逐条分歧**原样保留**：这是模型裁决分布，不是正确率，不是专家审核，不推广到其余 54 图。

- **wbv3_064（rv30_027）语义挂起**：模型判 usable，但其位置描述（含 location_clock=[3]）指向**右侧绿色合格区**，不是红色缺陷位置；下游若作"缺陷主位置"使用会错配。两份模型原答不改，监督使用前必须把"描述对象"与"缺陷位置"分开处理。
- **证据口径修正**：v1 终验以文件存在计 actually_viewed、以 mtime 序证打开顺序，均不充分。v2 证据=客户端审计日志：30/30 图 PNG Read 恰 1 次、旧 raw Read 恰 1 次、Write(observation) 早于 Read(旧raw)；started 写入先于 PNG Read；父会话零次读图。仍属客户端工具事件证据，后端请求数与模型上下文不可证。
- **自报时间失真**：Codex 点名 16 图核实成立（手写估计/本地钟面误标Z）；片4 整片自报时间同为本地误标Z（追加发现）。执行时刻一律以审计日志事件时刻为准；详见 追加记录/时间顺序矛盾核查.md。
- **wbv3_056**：哈希差异以程序difflib结果为准（单处插入、长度3），旧更正说明中"缺92两字符"表述不准；旧started与旧更正说明原件未改。
- 逐项程序结果与日志索引见本副本所在 追加记录_20261011_0133/ 目录。
"""
(OUT / "30图复核结果一览_v2.md").write_text(v1_md + footer, encoding="utf-8")
print("written:", OUT / "final_summary_v2.json")
print("written:", OUT / "30图复核结果一览_v2.md")
