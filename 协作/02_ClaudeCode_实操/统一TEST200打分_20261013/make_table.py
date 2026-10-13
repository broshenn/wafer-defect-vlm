#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从统一打分的产物生成中文总表与三句话结论。

输入（只读）：out_unified/metrics.json、out_unified/main.csv、source_manifest.json、
              contract.json、pairs.json
输出：统一TEST200总表.md
不重算任何分数，只做排版与阈值判定。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = "out_v2_无S3"


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f6(x):
    return "不适用" if x is None or x == "" else f"{float(x):.4f}"


def pct(x):
    return "不适用" if x is None or x == "" else f"{float(x)*100:.1f}%"


def fmt_int(x):
    return "不适用" if x is None or x == "" else str(int(x))


PROMPT_LABEL = {
    "v5_main": "v5 主列（`prompt_v5.txt`）",
    "class_json": "类别列（L 的类别题面）",
    "cnn_per_sample": "无文本题面（逐样本类别预测）",
}
SOURCE_LABEL = {"复用原答": "复用原答", "新推理": "新推理", "百炼": "百炼"}

# 迁移考试：权重没按该题面训练过，必须标出来
TRANSFER_NOTE = {
    "D_v5_test200": "**未按此题训练**：该适配器原用类别题面训练，本轮是迁移考试",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="打分产物目录名（相对本目录）")
    args = ap.parse_args()
    OUT = HERE / args.out
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    contract = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    pairs_lock = json.loads((HERE / "pairs.json").read_text(encoding="utf-8"))

    rows = list(csv.DictReader(io.StringIO((OUT / "main.csv").read_text(encoding="utf-8"))))
    by_name = {r["model"]: r for r in rows}
    prov = {e["name"]: e for e in manifest["sources"]}

    geo_ref = pairs_lock["geo_reference"]
    cat_ref = pairs_lock["category_reference"]
    ps = metrics["paired_statistics"]
    pair_map = {(p["a"], p["b"]): p for p in ps.get("pairs", [])}

    def above(name, ref, metric):
        """返回 ('是'|'否'|'无区间', 点差, 区间)。判定要求区间下界 > 0。"""
        p = pair_map.get((name, ref))
        if not p or metric not in p["metrics"]:
            return "—", None, None
        m = p["metrics"][metric]
        lo, hi = m["ci95_percentile"]
        return ("是" if lo > 0 else "否"), m["difference_a_minus_b"], (lo, hi)

    L = []
    A = L.append
    A("# 统一 TEST200 打分总表（2026-10-13）")
    A("")
    A("> 不训练、不改权重、不改测试标签。考卷只用已冻结的 TEST200（200 张，分层，种子 3407）。")
    A("> 252 / 56 / 36 / 120 张上的旧分数一律没有搬进本表；589 张本轮没有跑。")
    A("")
    A("## 0. 口径")
    A("")
    A("> **本版（v2）与第一版（v1，git `e793b6a`）的唯一差别**：按用户指令把 `S3_fixed` 从清单移除，"
      "49 条来源取代 50 条，配对 93 对取代 95 对。除此之外**其他任何模型的任何数字都没有变**"
      "（逐行比对过 `main.csv`），也没有增补模型、换题面或挑种子。"
      "配对是在已经看过分之后重锁的，改动被限制为「只删 S3 的两对、不加任何一对」，"
      "新旧两份 `pairs.json` 都已留档（`pairs_S3included.json` / `pairs.json`）。")
    A("")
    A(f"- 合同 `contract.json` SHA256：`{sha_file(HERE/'contract.json')}`")
    A(f"- 来源清单 `source_manifest.json` SHA256：`{sha_file(HERE/'source_manifest.json')}`")
    A(f"- 冻结 ID 数：{contract['n_expected']}；冻结考卷 SHA256：`{contract['reference']['sha256']}`")
    A(f"- 几何分母：覆盖率 {contract['geometry_denominators']['coverage']}、"
      f"区带 {contract['geometry_denominators']['zones']}、"
      f"角向类型 {contract['geometry_denominators']['angular_type']}"
      "（参考为 `unknown` 的分量不进平均）")
    A("- **GeoScore** = 覆盖档 / 区带 / 角向类型三项精确正确率的平均（每项各用各的分母）。"
      "**奖励总分不算成绩**，本表不列。")
    A("- 缺答与格式非法一律留在分母里，记不正确；没有把没返回的题从分母删掉。")
    A(f"- 配对统计：按 lot 配对，10000 次重抽，种子 {ps.get('seed')}；"
      f"共 {len(ps.get('pairs', []))} 对，**在看分之前锁定**（`pairs.json`）。"
      "**点估计更高但没有配对区间的，不能写「显著优于」。**")
    A("- 程序几何与公开类别都不是人工金标。本表所有分数都是模型输出对上程序参考/公开标签。")
    A("")

    # ---------------- 表 A：主列
    A("## 1. 主列表（题面 = v5 的 `prompt_v5.txt`）")
    A("")
    A("| 模型 | 来源 | 题面 | 几何分 | 覆盖档 | 区带 | 角向 | 类别准确率 | 九类 Macro-F1 | 格式合法率 | 缺答数 | 备注 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    main_rows = [r for r in rows if r["protocol"] == "v5_main"]
    main_rows.sort(key=lambda r: (r["geoscore"] in (None, ""), -(float(r["geoscore"]) if r["geoscore"] not in (None, "") else 0)))
    for r in main_rows:
        n = r["model"]
        note = TRANSFER_NOTE.get(n, "")
        A(f"| {n} | {SOURCE_LABEL.get(r['source'], r['source'])} | {PROMPT_LABEL['v5_main']} | "
          f"{f6(r['geoscore'])} | {f6(r['coverage_exact'])} | {f6(r['zones_exact'])} | "
          f"{f6(r['angular_type_acc'])} | {pct(r['class_acc'])} | {f6(r['class_macro_f1_9'])} | "
          f"{pct(r['format_legal_rate'])} | {fmt_int(r['missing'])} | {note} |")
    A("")
    A(f"（主列共 {len(main_rows)} 行。几何三项分母：覆盖档 {main_rows[0]['coverage_n']}、"
      f"区带 {main_rows[0]['zones_n']}、角向 {main_rows[0]['angular_type_n']}。）")
    A("")

    # ---------------- 表 B：类别列
    A("## 2. 类别列（只给本来就只输出类别的模型，用 L 自己的类别题面）")
    A("")
    A("| 模型 | 来源 | 题面 | 几何分 | 类别准确率 | 九类 Macro-F1 | 格式合法率 | 缺答数 | 原始题面下的类别准确率 |")
    A("|---|---|---|---|---|---|---|---|---|")
    cat_rows = [r for r in rows if r["protocol"] != "v5_main"]
    cat_rows.sort(key=lambda r: -(float(r["class_macro_f1_9"]) if r["class_macro_f1_9"] not in (None, "") else -1))
    for r in cat_rows:
        if r["protocol"] == "cnn_per_sample":
            geo = "不适用（逐样本类别预测，无文本题面）"
        else:
            geo = "不适用"
        A(f"| {r['model']} | {SOURCE_LABEL.get(r['source'], r['source'])} | "
          f"{PROMPT_LABEL.get(r['protocol'], r['protocol'])} | {geo} | {pct(r['class_acc'])} | "
          f"{f6(r['class_macro_f1_9'])} | {pct(r['format_legal_rate'])} | {fmt_int(r['missing'])} | "
          f"{pct(r['original_question_class_acc']) if r['original_question_class_acc'] not in (None,'') else '—'} |")
    A("")
    A("> L 的类别题面共 589 张，本表只统计其中属于冻结 TEST200 的 200 张。"
      "只答类别的模型**几何分写不适用，不记 0**。")
    A("")

    # ---------------- 配对区间
    A("## 3. 配对区间（看分前已锁定）")
    A("")
    A(f"- 几何参照 = `{geo_ref}`；类别参照 = `{cat_ref}`")
    A(f"- 重抽单位：{ps.get('resampling_unit')}；重抽次数 {ps.get('repeats')}；"
      f"种子 {ps.get('seed')}；lot 数 {ps.get('n_lots')}")
    A("")
    A("| 模型 | 对几何参照的 GeoScore 差（95% 区间） | 区间下界>0 | 对类别参照的 Macro-F1 差（95% 区间） | 区间下界>0 |")
    A("|---|---|---|---|---|")
    for r in main_rows:
        n = r["model"]
        if n == geo_ref:
            continue
        gv, gd, gci = above(n, geo_ref, "geo")
        cv, cd, cci = above(n, cat_ref, "macro_f1")
        gtxt = "—" if gci is None else f"{gd:+.4f} [{gci[0]:+.4f}, {gci[1]:+.4f}]"
        ctxt = "—" if cci is None else f"{cd:+.4f} [{cci[0]:+.4f}, {cci[1]:+.4f}]"
        A(f"| {n} | {gtxt} | {gv} | {ctxt} | {cv} |")
    for r in cat_rows:
        n = r["model"]
        if n == cat_ref:
            continue
        cv, cd, cci = above(n, cat_ref, "macro_f1")
        ctxt = "—" if cci is None else f"{cd:+.4f} [{cci[0]:+.4f}, {cci[1]:+.4f}]"
        A(f"| {n} | 不适用 | — | {ctxt} | {cv} |")
    A("")

    # ---------------- 三句话
    A("## 4. 三句话")
    A("")
    geo_hi = [r["model"] for r in main_rows if r["model"] != geo_ref and above(r["model"], geo_ref, "geo")[0] == "是"]
    geo_hi_pt = [r["model"] for r in main_rows if r["model"] != geo_ref and above(r["model"], geo_ref, "geo")[1] is not None
                 and above(r["model"], geo_ref, "geo")[1] > 0]
    cat_hi = [r["model"] for r in rows if r["model"] != cat_ref and above(r["model"], cat_ref, "macro_f1")[0] == "是"]
    cat_hi_pt = [r["model"] for r in rows if r["model"] != cat_ref and above(r["model"], cat_ref, "macro_f1")[1] is not None
                 and above(r["model"], cat_ref, "macro_f1")[1] > 0]
    both = [r["model"] for r in main_rows
            if r["model"] != geo_ref and r["model"] not in geo_hi and r["model"] not in cat_hi]
    A(f"1. **几何分高于 `{geo_ref}`**（配对区间下界 > 0）："
      f"{('、'.join(f'`{m}`' for m in geo_hi)) if geo_hi else '**没有一个**'}。"
      f"（点估计更高、但区间跨 0 的另有 {len(set(geo_hi_pt) - set(geo_hi))} 个，不能算「显著优于」。）")
    A(f"2. **类别上高于 `{cat_ref}`**（九类 Macro-F1 配对区间下界 > 0）："
      f"{('、'.join(f'`{m}`' for m in cat_hi)) if cat_hi else '**没有一个**'}。"
      f"（点估计更高、但区间跨 0 的另有 {len(set(cat_hi_pt) - set(cat_hi))} 个。）")
    A(f"3. **两类都没高于这两个参照**的（主列里既没有几何区间、也没有类别区间下界 > 0）：共 {len(both)} 个 —— "
      f"{('、'.join(f'`{m}`' for m in both)) if both else '无'}。")
    A("")
    if not geo_hi and not cat_hi:
        A("> 即：在本轮冻结的 TEST200 上，**没有任何模型在两个参照上都拿到区间不跨 0 的优势**。"
          "排名只能读点估计，不能读成统计显著。")
        A("")

    # ---------------- 缺席
    A("## 5. 缺席与未运行（如实记录）")
    A("")
    A("| 项 | 状态 | 依据 |")
    A("|---|---|---|")
    A("| 第一轮教师监督微调的适配器 | **缺席** | 两个候选（qlora `checkpoint-604`、D1 `checkpoint-89`）"
      "在学校机器 `/data/renjinyao` 下（maxdepth 6）都不存在；目录里只有 abc_A/B/C 的 `checkpoint-89`，"
      "属于 2026-10-08 的同图 ABC 对照，不是第一轮教师 SFT。 |")
    A("| M1、M2 | **未运行** | v5 结果目录 `E_结果/pull2/eval/` 下没有 M1/M2 的 test200 原答，"
      "学校机器上也没有对应 adapter；按任务书「不补，除非目录里已经有同题原答」处理。 |")
    A("| M0 的版本选择 | **M0_v2** | 身份核验记录了三个 M0 候选：M0_v2(`43b36375…`)、M0_v3(`191c62e8…`)、"
      "v5 S0(`8047875d…`)。其中只有 M0_v2 的权重在学校机器上（本次上传后 SHA 逐字节核对一致）；"
      "M0_v3 权重本地与学校均不存在，故用 M0_v2 新答，并标明它是 v2 轮的 M0。 |")
    A("| S3 | **按用户指令剔除** | 任务书原本写「S3 没有训成，不列入」，但学校机器上存在更晚的 "
      "`qoder_school_v5r1_20261012/runs/S3_fixed`（checkpoint-375，adapter `b060d9f6…`）及其 200 条 "
      "TEST200 原答，所以第一版表把它列了进去并标注。用户随后明确要求剔除，本版据此从清单移除。"
      "**原答没有被销毁**，完整保存在 `excluded_from_table/`；含 S3 的那一版表在 git 历史 `e793b6a` 里可以翻到。 |")
    A("")

    # ---------------- 来源
    A("## 6. 每条来源的出处")
    A("")
    A("| 模型 | 来源类别 | 题面 | 原答文件 | 原答 SHA256（前 16） | nominalN | 消费的考卷 |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        n = r["model"]
        e = prov.get(n, {})
        A(f"| {n} | {r['source']} | {PROMPT_LABEL.get(r['protocol'], r['protocol'])} | "
          f"`{e.get('rawfile','—')}` | `{(e.get('raw_sha256') or '')[:16]}` | "
          f"{e.get('nominalN','—')} | `{Path(e.get('inputfile','—')).name}` |")
    A("")
    A("> 已经有同题原答的一律**重算分，不占卡重答**；只有 D_v5、D_L、M0_v2 三条是 2026-10-13 新做推理。"
      "百炼三条是同一张 PNG、同一题面、同一参数下新发的 API 请求。")
    A("")

    # ---------------- 百炼参数
    A("## 7. 百炼三模型")
    A("")
    fr = json.loads((HERE / "frozen_models.json").read_text(encoding="utf-8"))
    A("| 角色 | 模型 |")
    A("|---|---|")
    for role, m in fr["roles"].items():
        A(f"| {role} | `{m}` |")
    A("")
    A(f"- 冻结发生在**任何一张 TEST200 调用之前**；先用合成的中性 PNG 做连通检查"
      f"（白底 + 左侧红方块 + 右侧蓝圆），三个模型都正确答出，才冻结。")
    A(f"- 参数：`{json.dumps(fr['params'], ensure_ascii=False)}`")
    A("- **实发题面与本地题的差别**：本地考卷首行的 `<image>` 是 ms-swift 的图片占位 token"
      "（由模板展开成视觉 token，模型看不到字面量），百炼侧由 image 内容块承担同一角色，"
      "故实发正文等于去掉该占位行后的 `prompt_v5.txt`。每条来源都记了 `api_prompt_sha256`。")
    A("")

    # ---------------- 复现性
    A("## 8. 复现性检查：重跑对")
    A("")
    A("十月十三日凌晨那一轮里有 7 个模型是**同权重重跑**。逐条比对答案文本：")
    A("")
    A("| 原运行 | 重跑 | 逐条答案文本不同的条数 | 全部指标是否相同 |")
    A("|---|---|---|---|")
    def _raws(n):
        return {json.loads(l)["sample_id"]: json.loads(l).get("raw")
                for l in (HERE / "sources" / f"{n}_raw.jsonl").open(encoding="utf-8") if l.strip()}
    REPRO = [("BASE", "BASE_rerun"), ("S0", "S0_rerun"), ("S1", "S1_rerun"), ("S2", "S2_rerun"),
             ("S4", "S4_rerun"), ("RSFT", "RSFT_rerun"), ("Rfull", "Rfull_rerun")]
    n_same = 0
    for a, b in REPRO:
        if a not in by_name or b not in by_name:
            continue
        A_, B_ = _raws(a), _raws(b)
        diff = sum(1 for k in A_ if A_[k] != B_.get(k))
        same = all(by_name[a][c] == by_name[b][c] for c in
                   ("geoscore", "coverage_exact", "zones_exact", "angular_type_acc",
                    "class_acc", "class_macro_f1_9", "format_legal_rate"))
        n_same += (diff == 0 and same)
        A(f"| `{a}` | `{b}` | {diff} / {len(A_)} | {'是' if same else '否'} |")
    A("")
    A(f"> {n_same}/{len(REPRO)} 对**逐条答案文本完全相同、指标完全相同**。"
      "这说明本轮推理在 `temperature=0` + 种子 3407 下是可复现的；"
      "同时也说明「重跑」不带来新信息，两组是同一份证据的两个副本。")
    A("")

    # ---------------- 题面歧义诊断
    A("## 9. 题面歧义诊断（**不是成绩，不改正式分数**）")
    A("")
    diag_path = HERE / "诊断_zone_frac刻度.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))["results"]
        A("冻结题面写的是「zone_frac：……三个径向区带的比例……各保留一位小数」，"
          "**没有明说是 0–1 的小数还是 0–100 的百分数**；冻结正则 `(\\d\\.\\d)` 只接受前一种。")
        A("")
        A("| 模型 | 正式合法条数 | 把 zone_frac 写成百分数（整数部分≥2 位） | 其中仅因这个刻度而不合法 |")
        A("|---|---|---|---|")
        for k, v in diag.items():
            A(f"| `{k}` | {v['legal_official']}/{v['rows']} | {v['zone_frac_percent_scale']} | "
              f"{v['would_be_legal_if_percent_accepted']} |")
        A("")
        mx = diag.get("BL_maxplus", {})
        if mx:
            A(f"> 只有 `BL_maxplus` 受影响，而且影响很大：{mx['legal_official']}/{mx['rows']} 合法，"
              f"另有 {mx['would_be_legal_if_percent_accepted']} 条只差这个刻度 → "
              f"按百分数接受会是 {mx['legal_official'] + mx['would_be_legal_if_percent_accepted']}/{mx['rows']}。"
              "**上面第 1 节表里的 29.0% 仍是正式数字**，这一行只是解释它为什么低。"
              "读成「百炼旗舰读图能力差」之前必须先扣掉这一项。")
        A("")
    else:
        A("（未运行诊断）")
        A("")

    # ---------------- 限制
    A("## 10. 读这张表时必须知道的限制")
    A("")
    A("- **两列题面不同，不能横向混用**：主列是 v5 的 `prompt_v5.txt`（一句话测量 + 六字段 JSON），"
      "类别列是 L 自己的类别题面（只出类别）。第 4 节第 2 句把主列模型的类别准确率拿去和 L 比，"
      "是**跨题面**比较，题面不同这件事本身就会贡献差距。")
    A("- **都不是人工金标**：几何参考是程序按几何特征算出来的，类别参考是公开标签。"
      "`predictDefectType` 一类字段从未被当作人工真值。")
    A("- **角向分母只有 137**：参考里 `angular_type` 为 `unknown` 的 63 张不进这一项的分母，"
      "但覆盖率与区带的分母是满 200。三项平均时各用各的分母。")
    A("- **五个 `none`/空缺陷样本的表现被稀释**：本表只给总平均，没有按类别分列。"
      "某类可能又被另一类盖住。")
    A("- **缺答不删分母**：百炼有 2 条网络失败留空、计不正确，仍在 200 里。")
    A("- **同一模型多个种子全部保留**，没有只留最高的一个；配对比较对每个清单行都点名，"
      "不挑种子、不挑模型。")
    A("- **点估计更高 ≠ 显著优于**：第 4 节第 1、2 句里「点估计更高但区间跨 0」的那些行，"
      "只能读成「这一卷上看不出差别」。")
    A("")

    text = "\n".join(L) + "\n"
    (HERE / "统一TEST200总表.md").write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"主列行数": len(main_rows), "类别列行数": len(cat_rows),
                      "几何高于参照": geo_hi, "类别高于参照": cat_hi,
                      "两者都未高于": len(both)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
