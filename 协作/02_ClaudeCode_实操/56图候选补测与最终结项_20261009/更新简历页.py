#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 56 图留出候选补测的结果并进简历页，**另存新版本**（不改旧文件）。

只做定点文本替换，工具条 / CSS / 交互脚本原样保留。
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "协作/01_Codex_指挥/自动接续_20261009/最终简历稿/项目经历页_可编辑.html"
DST = Path(__file__).resolve().parent / "项目经历页_56图版.html"

s = SRC.read_text(encoding="utf-8")
orig = s


def swap(old: str, new: str, note: str):
    global s
    assert s.count(old) == 1, f"锚点不唯一/不存在（{note}）：{s.count(old)}"
    s = s.replace(old, new)
    print(f"  改：{note}")


# ---- 1. 第一段的「实测结果」改成以留出候选打头 ----
swap(
    "<li><strong>实测结果：</strong>同基座只把监督从「无」换成类别标签，准确率 "
    "<strong>0.1667 → 0.8889</strong>；84 张开发验证集上 <strong>77/84（91.7%）</strong>，"
    "九类 Macro-F1 <strong>0.9227</strong>。",
    "<li><strong>实测结果（56 张留出候选：与训练池的 ID / lot / PNG 内容交集均为 0）：</strong>"
    "同基座只把监督从「无」换成类别标签，准确率 <strong>0.2679 → 0.8214</strong>"
    "（同图配对 bootstrap 区间 <strong>[0.411, 0.696]，不含 0</strong>）；"
    "描述监督的 <code class=\"code\">D</code> 为 <strong>0.7679</strong>，"
    "与 <code class=\"code\">L</code> 之差 <strong>[−0.054, +0.161] 跨 0 —— 本轮分不出高下</strong>。"
    "开发集 36 图上同基座对照 <strong>0.1667 → 0.8889</strong>，"
    "84 张开发验证集 <strong>77/84（91.7%）</strong>。",
    "第一段实测结果（以 56 图留出候选打头）",
)

# ---- 2. 负结果追加 56 图上的新发现 ----
swap(
    "（3）<code class=\"code\">L</code> 六维合计垫底<strong>完全由「格式」一维造成</strong>，"
    "去掉格式后其视觉四维反好于 <code class=\"code\">D</code>，准确说法是「视觉内容中上、格式合规最差」。</li>",
    "（3）<code class=\"code\">L</code> 六维合计垫底<strong>完全由「格式」一维造成</strong>，"
    "去掉格式后其视觉四维反好于 <code class=\"code\">D</code>，准确说法是「视觉内容中上、格式合规最差」；"
    "<strong>56 图留出候选上这一现象复现</strong>（<code class=\"code\">L</code> 视觉四维最差项最少 12/224、"
    "格式却坏 <strong>51/56</strong>，仍是把 JSON <code class=\"code\">null</code> 写成字符串 "
    "<code class=\"code\">\"null\"</code>）；"
    "（4）<strong>格式违规会压低类别分</strong>：<code class=\"code\">L</code> 唯一一条 schema 不合法里"
    "类别其实答对（<code class=\"code\">Scratch</code>，真值也是它），被 "
    "<code class=\"code\">radial_zone</code> 越界拖成「未解析」——严格口径 0.8214、宽松口径 0.8393；"
    "（5）56 张里<strong>至少 4 张公开标签与图上明显不符</strong>（例：标 Center 的图除最外圈外几乎全红），"
    "<strong>「答错」里含标签噪声</strong>。</li>",
    "负结果追加 56 图三条",
)

# ---- 3. 个人边界：把「开发集而非盲测卷」升级为两段式说明 ----
swap(
    "<strong>描述结论是模型自查，不是人工金标</strong>；36 图是<strong>开发集而非盲测卷</strong>；",
    "<strong>描述结论是模型自查，不是人工金标</strong>；"
    "36 图是<strong>开发集</strong>，56 图是<strong>可见历史隔离后的留出候选</strong>"
    "（<strong>七类、缺 Near_full / none，完整历史接触未核，仍不是独立九类盲测卷</strong>）；",
    "个人边界：两套图集的性质分开写",
)

# ---- 4. HTTP 段追加 56 图真实推理 ----
swap(
    "<li><strong>实测结果：</strong>固定 36 图<strong>真实 HTTP 回放</strong>",
    "<li><strong>实测结果：</strong>租用单卡顺序完成 <code class=\"code\">Base</code> / "
    "<code class=\"code\">L</code> / <code class=\"code\">D</code> × 56 图 = <strong>168 条真实推理，"
    "0 失败</strong>，单卡占用 <strong>541 秒</strong>（9.02 设备分钟），峰值显存约 <strong>19.1 GiB</strong>，"
    "释放后 1 MiB、无残留进程；每条先写 started 台账、再存原答与四项哈希（图片 / 题面 / 检查器 / adapter）。"
    "固定 36 图<strong>真实 HTTP 回放</strong>",
    "HTTP 段追加 168 条真实推理",
)

# ---- 5. 对照设计：把 56 图补测单列一条 ----
swap(
    "冻结题面 sha256、固定解码参数，日志留痕可复算。</li>",
    "冻结题面 sha256、固定解码参数，日志留痕可复算。</li>\n"
    "          <li class=\"nested\"><strong>留出候选补测：</strong>"
    "另用 <strong>56 张</strong>与训练池 ID / lot / PNG 内容交集均为 0 的图，"
    "在同基座、同题面、同解码参数下补测 <code class=\"code\">Base</code> / <code class=\"code\">L</code> / "
    "<code class=\"code\">D</code>，并做<strong>同图配对差与 10000 次按图 bootstrap</strong>（seed 3407）；"
    "复核时把「视觉四维」与「格式」<strong>分列</strong>，不合成单一描述准确率。</li>",
    "对照设计后新增留出候选补测条",
)

# ---- 6. 技能条目微调 ----
swap(
    "<strong>评测方法：</strong>冻结枚举 schema 设计　·　同基座对照　·　Macro-F1　·　置信区间　·　解析口径分层（严格 / 围栏诊断）",
    "<strong>评测方法：</strong>冻结枚举 schema 设计　·　同基座对照　·　"
    "留出集隔离（ID / lot / 图像内容三重交集为 0）　·　同图配对差与按图 bootstrap　·　"
    "Macro-F1　·　解析口径分层（严格 / 围栏诊断）",
    "技能：补留出集隔离与配对检验",
)
swap(
    "<strong>数据：</strong>WM811K 抽样与分层　·　lot 级隔离　·　图片内容级去重　·　候选质量分账",
    "<strong>数据：</strong>WM811K 抽样与分层　·　lot 级隔离　·　图片内容级去重　·　"
    "留出候选考卷冻结（题面 / 检查器 / 逐图 SHA 冻结与上传前核对）　·　候选质量分账",
    "技能：数据行补考卷冻结",
)

assert s != orig
DST.write_text(s, encoding="utf-8")
print(f"\n写出 {DST.name}  {len(s)} 字符（原 {len(orig)}）")

# 自检：无外链、无待补之外的个人信息、新数字已进
bad = re.findall(r'(?i)(https?://(?!www\.w3\.org)[^\s"\']+|sk-[A-Za-z0-9]{15,})', s)
print("外链/密钥自检：", bad if bad else "无")
for k in ["0.8214", "0.7679", "0.2679", "168 条真实推理", "541 秒", "51/56", "0.8393"]:
    print(f"  含「{k}」：", k in s)
print("  待补 出现次数：", s.count("待补"))
