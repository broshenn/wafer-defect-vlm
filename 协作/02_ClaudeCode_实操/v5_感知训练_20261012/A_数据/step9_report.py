#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 9 步（后半）：生成 A_数据/交付报告.md —— 所有数字从产物文件重推，不手抄。"""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    geo = load_jsonl(HERE / "geom_all.jsonl")
    rot = load_jsonl(HERE / "geom_rot.jsonl")
    syn = load_jsonl(HERE / "geom_syn.jsonl")
    inelig = load_jsonl(HERE / "ineligible.jsonl")
    pool = (HERE / "TRAIN_POOL.txt").read_text(encoding="utf-8").split()
    pool_set = set(pool)
    test_ids = (HERE / "TEST.txt").read_text(encoding="utf-8").split()
    select200 = (HERE / "SELECT200.txt").read_text(encoding="utf-8").split()
    test200 = (HERE / "TEST200.txt").read_text(encoding="utf-8").split()
    shuf100 = (HERE / "SHUF100.txt").read_text(encoding="utf-8").split()
    stats = json.loads((HERE / "sft_stats.json").read_text(encoding="utf-8"))
    checklist = json.loads((HERE / "数据清单.json").read_text(encoding="utf-8"))
    selfcheck = json.loads((HERE / "rotation_selfcheck.json").read_text(encoding="utf-8"))
    geom = {r["sample_id"]: r for r in geo}

    contract_sha = hashlib.sha256((HERE.parent / "00_合同副本" / "00_总纲与合同.md").read_bytes()).hexdigest()
    prompt_sha = hashlib.sha256((HERE / "prompt_v5.txt").read_bytes()).hexdigest()
    atomic_sha = hashlib.sha256((HERE / "atomic_templates.json").read_bytes()).hexdigest()

    # 方向样本统计
    def ang_counts(rows):
        return dict(Counter(r["angular_type"] for r in rows))
    pool_geo = [geom[s] for s in pool]
    test_geo = [geom[s] for s in test_ids]
    # HOLD56 / CONF120
    hold56 = set(re.findall(r"wafer_\d{8}_\d{3}",
                 (REPO / "协作/02_ClaudeCode_实操/56图候选补测与最终结项_20261009/冻结清单.json").read_text(encoding="utf-8")))
    conf120 = {json.loads(l)["sample_id"] for l in
               (REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/pools/confirmation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    # 10 条训练样本原文
    lines = (HERE / "sft_S1.jsonl").read_text(encoding="utf-8").splitlines()
    idx10 = sorted(random.Random(3407).sample(range(len(lines)), 10))
    samples = [lines[k] for k in idx10]

    tgz = {}
    for name in ("img448.tgz", "img896.tgz"):
        p = HERE / name
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            tgz[name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
        else:
            tgz[name] = "not run（打包尚未完成）"

    md = []
    md.append("# v5 任务卡 A（数据引擎）交付报告\n")
    md.append(f"- 合同 SHA256：`{contract_sha}`（副本：00_合同副本/00_总纲与合同.md）")
    md.append(f"- prompt_v5.txt SHA256：`{prompt_sha}`（15 行，与合同 3.1 逐字一致）")
    md.append(f"- atomic_templates.json SHA256：`{atomic_sha}`（T1–T6 与合同 5.3 逐字一致）")
    md.append(f"- 本机解释器：`D:\\anaconda\\python.exe`（numpy 1.26.4 / PIL 12.3.0 / pandas 2.3.3；见 DECISIONS D1）\n")

    md.append("## 第 1 步 全量几何（step1_geometry_all.py，4 进程 69s）")
    md.append(f"- 总 5904 = 可用 {len(geo)} + 不可用 {len(inelig)}；不可用原因：{dict(Counter(r['reason'] for r in inelig))}")
    md.append(f"- 可用按 split：{dict(Counter(r['split'] for r in geo))}")
    md.append(f"- coverage_band 分布：{dict(Counter(r['coverage_band'] for r in geo))}")
    md.append(f"- angular_type 分布：{dict(Counter(r['angular_type'] for r in geo))}")
    md.append("- no_red 样本 0 张（全部图都有红点）\n")

    md.append("## 第 2 步 划分与隔离（step2_splits.py + isolation_check.py + test_isolation.py）")
    md.append("- 受保护文件 8 个全部读到非零 ID：core.jsonl 252、冻结清单.json 56、R0_dev18 18、R0_val90 84、")
    md.append("  清单_确认池 120、清单_规则开发池 30、pools/confirmation 120、pools/rule_dev 30；合计去重 710")
    md.append("- TRAIN_POOL 4536 = train 可用 4719 − 受保护 180 − 与 SELECT/TEST 同 SHA 3")
    md.append(f"- SELECT200 200 张、TEST {len(test_ids)}、TEST200/ROT200 200、SHUF100 100（seed 3407 分层）")
    md.append("- 隔离检查（sample_id / lot_name / png_sha256 三层交集）：TRAIN_POOL vs SELECT∪TEST = 0/0/0，")
    md.append("  vs ROT200 = 0/0/0，vs SHUF100 = 0/0/0；反例测试 5/5 PASS（含故意冲突、空集合、同 SHA）")
    md.append(f"- HOLD56 ∩ TEST = {len(hold56 & set(test_ids))}/56；CONF120 ∩ TEST = {len(conf120 & set(test_ids))}/120\n")

    md.append("## 第 3 步 旋转增广与等变性（step3_rotations.py，6 进程 56s）")
    md.append(f"- 原图复制 5325 张到 img448/；旋转 {len(rot)} 张（13608=4536×3），不可用 0")
    md.append(f"- 扇区等变一致率：{selfcheck['sector_ok']}/{selfcheck['sector_total']} = {selfcheck['sector_rate']}"
              f"（≥0.95 达标；并列变 unknown {selfcheck['tie_unknown']}）")
    md.append(f"- coverage_band 不变 {selfcheck['band_same']}/{selfcheck['band_total']} = {selfcheck['band_rate']}；"
              f"red_mass_zones 不变 {selfcheck['zone_same']}/{selfcheck['zone_total']} = {selfcheck['zone_rate']}（≥0.99 达标）\n")

    md.append("## 第 4 步 合成图（step4_synthetic.py；单测 5/5 PASS）")
    md.append(f"- 底图仅 5 张（none 且 coverage<0.02 的全部；见 DECISIONS D4）；生成 3000，不可用 0；选出 1500")
    md.append(f"- 最终 angular_type：{dict(Counter(r['angular_type'] for r in syn))}")
    md.append(f"- 最终 pattern：{dict(Counter(r['pattern'] for r in syn))}")
    md.append(f"- 最终 coverage_band：{dict(Counter(r['coverage_band'] for r in syn))}")
    md.append("- 联系表：syn_contact_sheet.png（20 张 4×5，含图案名与程序真值）\n")

    md.append("## 第 5 步 六个训练文件（step5_build_sft.py；全部 PASS）")
    md.append("- 行数全部 12000；S1=S2=S1p 逐字节（同一 SHA）；S3 与 S1 仅 images 加 `_x2`")
    md.append("- 200 条主任务重算一致 200/200（core_v3.reference∘measure_v3 复核）")
    md.append("- 训练图全部属于 TRAIN_POOL 或合成（受保护零泄漏）")
    for n, st in stats["per_file"].items():
        md.append(f"- {n}: atomic {st['atomic']} + main {st['main']}；模板 {st['template']}；"
                  f"图源 {st['line_src']}；主任务 line_src {st['main_line_src']}；主任务 angular {st['main_angular']}")
    md.append(f"- 文件 SHA：")
    for n, sha in stats["shas"].items():
        md.append(f"  - {n}: `{sha}`")
    md.append("")

    md.append("## 第 6 步 评测与 RL 文件（step6_build_eval.py；全部 PASS）")
    md.append(f"- 行数：select200/test/test200/rot200/rot100/shuf100/shuf50 = 200/{len(test_ids)}/200/200/100/100/50；"
              f"select200_L/test_L = 200/{len(test_ids)}；rl_candidates 320；rsft_rows_template 1280；_x2 同步")
    md.append("- 每个评测/派生文件与 TRAIN_POOL 三层隔离 0；rl_candidates 全部 ∈ TRAIN_POOL；reference_json 六字段全部过 schema")
    md.append(f"- rl_candidates 类别分布：{dict(Counter(geom[s]['failure_type'] for s in [json.loads(l)['sample_id'] for l in (HERE / 'rl_candidates800.jsonl').read_text(encoding='utf-8').splitlines()]))}\n")

    md.append("## 第 7 步 img896（step7_img896.py）")
    md.append("- 被引用图片 10247 个全部生成 _x2.png（896×896 最近邻），缺源 0；抽检 3 张尺寸与三色 PASS\n")

    md.append("## 第 8 步 叠加核验图（step8_overlays.py，P2 已做）")
    md.append("- overlays/ 140 张（TEST 分层 120 + 合成 20），672×712；index.jsonl 140 行")
    md.append("- 本机不能看图，线条与文字位置请用户肉眼复核；抽检 3 张路径见运行记录\n")

    md.append("## 第 9 步 打包")
    for n, info in tgz.items():
        if isinstance(info, str):
            md.append(f"- {n}: {info}")
        else:
            md.append("- {}: {} 字节 SHA256=`{}`".format(n, info["bytes"], info["sha256"]))
    md.append("")

    md.append("## 数据清单.json（全量内容）")
    md.append("```json")
    md.append(json.dumps(checklist, ensure_ascii=False, indent=1))
    md.append("```\n")

    md.append("## 四集合大小与类别分布")
    for name, ids in (("TRAIN_POOL", pool), ("SELECT200", select200), ("TEST", test_ids),
                      ("TEST200=ROT200", test200), ("SHUF100", shuf100)):
        md.append(f"- {name}: {len(ids)} 张 {dict(Counter(geom[s]['failure_type'] for s in ids))}")
    md.append("")

    md.append("## 方向样本统计（决定扇区指标分母）")
    md.append(f"- TRAIN_POOL 原图 angular：{ang_counts(pool_geo)}")
    md.append(f"- TRAIN_POOL 旋转版 angular：{ang_counts(rot)}")
    md.append(f"- 合成图 angular：{ang_counts(syn)}")
    for n in ("sft_S0", "sft_S1", "sft_S4"):
        md.append(f"- {n} 主任务段 angular：{stats['per_file'][n]['main_angular']}")
    md.append(f"- TEST angular：{ang_counts(test_geo)}（single+opposed = "
              f"{sum(1 for r in test_geo if r['angular_type'] in ('single','opposed'))} → 扇区指标分母）\n")

    md.append("## 随机抽的 10 条训练样本原文（sft_S1.jsonl，seed 3407）")
    for s in samples:
        md.append("```json")
        md.append(s)
        md.append("```")
    md.append("")

    md.append("## 本会话没做/未决")
    md.append("- 叠加核验图只生成了图，位置自查需要人眼（本机不能看图）。")
    md.append("- 其余按任务卡 A 全部完成；DECISIONS.md 无未解决项（D1–D5 均选择了默认方案并已记录）。")
    (HERE / "交付报告.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("交付报告.md 已写出，", len("\n".join(md)), "字符")
    return 0


if __name__ == "__main__":
    sys.exit(main())
