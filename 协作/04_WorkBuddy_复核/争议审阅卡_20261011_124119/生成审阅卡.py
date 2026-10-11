# -*- coding: utf-8 -*-
"""争议审阅卡生成器（只整理现成产物：不读图、不新增标注/复核、不改任何原答与裁决）。

范围 = 候选分账机器核并集：review_or_reject(9) ∪ 绿色定位对象{018,064,082}，去重共12图。
每卡：ID/来源/PNG链接与现算SHA；Kimi原答与GLM观察/裁决原文摘录+指纹；争议字段与争议句
（定位对象红色失效/绿色合格/全局/未知分列）；审核者意见/确认范围/最终状态留空
（默认pending_review、gold_eligible:false）；操作顺序证据链接与未核实边界。
018/082 无本轮GLM复核，如实写"无"，不补造。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"D:/pycode/晶圆图研究")
BLIND = REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/WorkBuddy84/blind84.jsonl"
CODEXVER = REPO / "协作/01_Codex_指挥/接手续验收_20261011/WorkBuddy30_独立核验.json"
OLD = REPO / "协作/04_WorkBuddy_复核/方向形态84_v3_20261010_200950"
R30 = REPO / "协作/04_WorkBuddy_复核/内容复核30_v3_20261010_233400"
APPX = R30 / "追加记录_20261011_0133"
OUT = Path(__file__).parent

R2V = {"wbv3_002": "rv30_001", "wbv3_006": "rv30_004", "wbv3_009": "rv30_005",
       "wbv3_024": "rv30_009", "wbv3_029": "rv30_010", "wbv3_033": "rv30_012",
       "wbv3_034": "rv30_013", "wbv3_038": "rv30_015", "wbv3_061": "rv30_025",
       "wbv3_064": "rv30_027"}  # 018/082 无GLM复核

# 定位对象与争议要点（人工依据现成文本整理；原文仍以摘录为准，本表仅作归类索引）
CURATED = {
    "wbv3_002": {"kimi_obj": "红色失效（边缘整圈环带＋内部混杂，全图性断言）", "glm_obj": "全局（全局混杂散布、无主导结构）",
                 "note": "环状带强断言 vs 全局均匀混杂：图上是否构成连续环是核心分歧"},
    "wbv3_006": {"kimi_obj": "红色失效（右上+左下两团块）", "glm_obj": "红色失效（左下+右上两团块）",
                 "note": "两团相对大小写反：旧答右上(clock=2)较大，独立观察左下(clock≈8)较大"},
    "wbv3_009": {"kimi_obj": "红色失效（右上+左下两处小团）", "glm_obj": "红色失效（右上横向小群+左下紧凑块）",
                 "note": "左下块位置：旧答'偏内部' vs 独立观察'偏边缘'"},
    "wbv3_018": {"kimi_obj": "绿色合格（绿色合格点全局散布、下半略密）", "glm_obj": None,
                 "note": "Codex指认绿色定位对象：位置字段主语是绿色合格点；无GLM复核"},
    "wbv3_024": {"kimi_obj": "红色失效（右侧+右下边缘斜线群）", "glm_obj": "红色失效（右下密集团块+右缘细线群）",
                 "note": "延伸轴：旧答[11,5]，独立观察无法可靠判定单一轴（留空）"},
    "wbv3_029": {"kimi_obj": "红色失效（右下边缘聚集团块）", "glm_obj": "红色失效（右下细长斜线，轴1-7）",
                 "note": "形态定性相反：紧凑团块 vs 细长斜线"},
    "wbv3_033": {"kimi_obj": "红色失效（顶部12点团块+其余零散）", "glm_obj": "红色失效（顶部密集带+全局密布）",
                 "note": "顶部定位一致；争议在其余区域密度：旧答'稀疏/少量' vs 独立观察'中下部红点密布'"},
    "wbv3_034": {"kimi_obj": "红色失效（中心竖直带+横向短带呈十字）", "glm_obj": "红色失效（枝状结构：竖直主线+右上斜分支）",
                 "note": "十字定性 vs 枝状+右上分支（分支轴1-7）"},
    "wbv3_038": {"kimi_obj": "红色失效（中上部12点团块）", "glm_obj": "红色失效（中心偏右上团块，clock=2）",
                 "note": "位置钟点：旧答[12] vs 独立观察[2]，模糊结构判读分歧"},
    "wbv3_061": {"kimi_obj": "全局（全局散布、无主导结构）", "glm_obj": "红色失效（底部边缘6点密集带为主要结构）",
                 "note": "主要结构存在性：旧答'无明显主导' vs 独立观察'底部边缘密集带最突出'"},
    "wbv3_064": {"kimi_obj": "绿色合格（右侧3点绿色团块）", "glm_obj": "绿色合格（右侧3点绿色成片区）",
                 "note": "语义挂起：位置描述对象是绿色合格区，下游若当红色缺陷位置会错配；另红色覆盖范围存在粒度差异"},
    "wbv3_082": {"kimi_obj": "绿色合格（顶部+底部偏边缘两处绿色团块）", "glm_obj": None,
                 "note": "Codex指认绿色定位对象：位置字段主语是绿色主要结构；无GLM复核"},
}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    blind = {json.loads(l)["item_id"]: json.loads(l)
             for l in BLIND.read_text(encoding="utf-8").splitlines() if l.strip()}
    ver = json.loads(CODEXVER.read_text(encoding="utf-8"))
    alloc = ver["followup_allocation"]
    vermap = {p["item_id"]: p for p in ver["per_item"]}
    audit = json.loads((APPX / "程序A_审计日志时间线.json").read_text(encoding="utf-8"))
    auditmap = {r["item_id"]: r for r in audit["per_item"]}

    # 并集机器核（不自行扩范围）
    r9 = set(alloc["review_or_reject"]); green = {"wbv3_018", "wbv3_064", "wbv3_082"}
    union = sorted(r9 | green)
    assert len(union) == 12 and set(R2V) <= set(union) - {"wbv3_018", "wbv3_082"}

    cards = []
    for it in union:
        rid = R2V.get(it)
        cur = CURATED[it]
        b = blind[it]
        png = Path(b["image_path"])
        png_sha_now = sha256_file(png)
        rel_png = "../../../data/images/" + png.name
        kimi_raw_p = OLD / f"{it}.raw.txt"
        kimi_txt = kimi_raw_p.read_text(encoding="utf-8").strip()
        kimi_sha = sha256_file(kimi_raw_p)

        L = []
        L.append(f"# 审阅卡 · {it}（{b['sample_id']}）")
        tag = []
        if it in r9: tag.append("requires_review/reject（模型裁决）" if it not in ("wbv3_002", "wbv3_029") else "reject（模型裁决）")
        if it in green: tag.append("绿色定位对象（Codex指认）")
        if it == "wbv3_064": tag.append("语义挂起 semantic_hold_in_usable")
        L.append("")
        L.append(f"- **状态**：`pending_review`　**gold_eligible**：`false`")
        L.append(f"- **归类**：{'；'.join(tag)}")
        L.append(f"- **item_id**：{it}　**sample_id**：{b['sample_id']}　**GLM复核编号**：{rid or '无（本轮无GLM复核）'}")
        L.append(f"- **来源**：Kimi原答＝84图盲标轮（用户确认界面选择Kimi K3；真实后端未核实）；"
                 f"GLM观察/裁决＝内容复核30轮（会话声明GLM-5.3-Flash；真实后端未核实）。两轮均为AI产出，非人审gold。")
        L.append("")
        L.append("## 原图（仅供审核者查看；本轮整理未作任何新视觉判读）")
        L.append("")
        L.append(f"![{b['sample_id']}]({rel_png})")
        L.append("")
        L.append(f"- 文件：`{png}`")
        L.append(f"- 当前SHA256（本次现算）：`{png_sha_now}`")
        L.append(f"- 盲清单SHA256：`{b['image_sha256']}`　一致性：{'一致' if png_sha_now == b['image_sha256'] else '**不一致**'}")
        L.append("")
        L.append("## Kimi 原答（原文摘录，未改动）")
        L.append("")
        L.append("```json")
        L.append(kimi_txt)
        L.append("```")
        L.append("")
        L.append(f"- 原答文件：`{kimi_raw_p}`　SHA256（现算）：`{kimi_sha}`")
        L.append("")
        if rid:
            obs_p = R30 / f"{rid}.observation_raw.json"
            rev_p = R30 / f"{rid}.review.json"
            obs_txt = obs_p.read_text(encoding="utf-8").strip()
            rev = json.loads(rev_p.read_text(encoding="utf-8"))
            vp = vermap[it]
            L.append("## GLM 独立观察（原文摘录，未改动）")
            L.append("")
            L.append("```json")
            L.append(obs_txt)
            L.append("```")
            L.append("")
            L.append(f"- 观察文件：`{obs_p}`　SHA256（现算）：`{sha256_file(obs_p)}`")
            L.append("")
            L.append("## GLM 裁决与独立复算")
            L.append("")
            L.append(f"- 模型裁决 overall：**{rev['overall']}**（review文件 SHA256：`{sha256_file(rev_p)}`）")
            cf = vp.get("contradicted_fields") or []
            uf = vp.get("unverifiable_fields") or []
            L.append(f"- Codex独立复算 contradicted 字段：{('、'.join(cf)) if cf else '无'}")
            L.append(f"- Codex独立复算 unverifiable 字段：{('、'.join(uf)) if uf else '无'}")
            L.append("")
            if cf or uf:
                L.append("### 争议字段与具体争议句（判读证据原文摘录自 review.json）")
                L.append("")
                L.append("| 字段 | 判定 | 证据原文（含被审句） |")
                L.append("|---|---|---|")
                for f in cf:
                    c = rev["checks"].get(f)
                    if c: L.append(f"| {f} | contradicted | {c['evidence']} |")
                for f in uf:
                    c = rev["checks"].get(f)
                    if c: L.append(f"| {f} | unverifiable | {c['evidence']} |")
                L.append("")
            L.append("### 定位对象（模型位置描述的主语归类）")
            L.append("")
            L.append(f"- Kimi定位对象：{cur['kimi_obj']}")
            L.append(f"- GLM定位对象：{cur['glm_obj']}")
            L.append(f"- 争议要点：{cur['note']}")
            L.append("")
            sr = rev.get("suggested_revision") or ""
            L.append(f"- suggested_revision（**仅建议，不自动作正确答案**）：{sr if sr else '（无）'}")
            L.append("")
            a = auditmap[it]
            def t(lst): return lst[0]["utc"][11:19] + "Z" if lst else "—"
            L.append("### 独立观察操作顺序证据（客户端审计日志，机器时间UTC）")
            L.append("")
            L.append(f"- 事件序：started写 {t(a['started_write'])} → PNG读 {t(a['png_read'])}（{a['png_read_count']}次） → "
                     f"observation写 {t(a['observation_write'])} → 旧raw读 {t(a['oldraw_read'])}（{a['oldraw_read_tool_count']}次） → "
                     f"review写 {t(a['review_write'])} → done写 {t(a['done_write'])}")
            L.append(f"- 详见：`{APPX}/程序A_审计日志时间线.json`、`{APPX}/时间顺序矛盾核查.md`、`{APPX}/证据来源与声明降级.md`")
            L.append(f"- started/observation 自报时间存在失真（手写估计/本地误标Z），执行时刻以审计事件时刻为准。")
        else:
            L.append("## GLM 独立观察与裁决")
            L.append("")
            L.append("- **无**（wbv3_018/082 不在内容复核30轮名单内，本轮无GLM观察与裁决；按任务书如实写无，不补造。）")
            L.append("")
            L.append("### 当前争议（来源：Codex 指认绿色定位对象）")
            L.append("")
            L.append(f"- Kimi定位对象：{cur['kimi_obj']}")
            L.append(f"- GLM定位对象：无（无GLM复核）")
            L.append(f"- 争议要点：{cur['note']}")
            L.append(f"- Codex分账归属：{it} ∈ remaining_unreviewed（未复核54图之列），不在本轮候选副本20图内。")
            L.append("")
        L.append("## 审核者填写区（全部留空，待人审）")
        L.append("")
        L.append("| 项目 | 内容 |")
        L.append("|---|---|")
        L.append("| 审核者意见 | |")
        L.append("| 确认范围 | |")
        L.append("| 最终状态 | （默认 `pending_review`） |")
        L.append("")
        L.append("## 边界（不可省略）")
        L.append("")
        L.append("- 真实后端模型/采样参数、模型上下文内容、「首次认知未被改写」均**未核实**；观察先于旧答打开仅有客户端工具事件顺序证据。")
        L.append("- 本卡整理**未作新视觉判读**；模型裁决（含overall与suggested_revision）不等于正确答案，`gold_eligible=false`。")
        L.append("- 该图已进入开发/复核流程，**不得当留出测试图**使用。")
        L.append("")
        p = OUT / f"卡_{it}.md"
        p.write_text("\n".join(L), encoding="utf-8")
        cards.append((it, rid, p.name))

    # 索引
    idx = []
    idx.append("# 争议审阅卡索引（2026-10-11，只整理现成产物）")
    idx.append("")
    idx.append(f"生成：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}。范围=候选分账机器核并集："
               f"`review_or_reject`(9) ∪ 绿色定位对象{{wbv3_018, wbv3_064, wbv3_082}}，去重 **12图**；未自行扩范围。")
    idx.append("")
    idx.append("| 卡 | item_id | GLM复核编号 | 模型裁决 | 定位对象(Kimi/GLM) | 状态 |")
    idx.append("|---|---|---|---|---|---|")
    for it, rid, fn in cards:
        cur = CURATED[it]
        if rid:
            rev = json.loads((R30 / f"{rid}.review.json").read_text(encoding="utf-8"))
            overall = rev["overall"]
            glm = "无" if cur["glm_obj"] is None else cur["glm_obj"].split("（")[0]
        else:
            overall = "—（无GLM复核）"
            glm = "无"
        idx.append(f"| [{fn}]({fn}) | {it} | {rid or '—'} | {overall} | {cur['kimi_obj'].split('（')[0]} / {glm} | pending_review |")
    idx.append("")
    idx.append("## 使用规则")
    idx.append("")
    idx.append("- 每卡审核者意见/确认范围/最终状态**留空**待人审；默认 `pending_review`、`gold_eligible:false`；suggested_revision 仅为建议，不自动作正确答案。")
    idx.append("- 模型裁决分布（usable 21 / requires_review 7 / reject 2）不是正确率；本索引不计算、不更新任何正确率。")
    idx.append("- 卡内嵌图仅供查看，本轮整理未作新视觉判读；不新增图片标注/复核请求、不扩标其余图片。")
    idx.append("- 已复核图进入过开发/复核流程，**不得当留出测试图**。")
    idx.append("- 证据与边界：见 `内容复核30_v3_20261010_233400/追加记录_20261011_0133/`（审计日志时间线、时间矛盾核查、证据来源与声明降级）。")
    idx.append("- 客户端用量口径：本轮整理为纯本地文件操作，本会话工具层零图片读取请求（由审计日志核验，见 会话账目.md）；**不冒称整轮模型账号用量为零**。")
    (OUT / "00_索引.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"cards: {len(cards)}, index written")
    for it, rid, fn in cards:
        print(" ", fn, f"({rid or '无GLM'})")

if __name__ == "__main__":
    main()
