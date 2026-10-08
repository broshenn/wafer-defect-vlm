#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 段：固定**描述事实审阅集**，避免再用类别分替代描述能力（纯 CPU，不占 GPU）。

依据 `任务书_20261008_ClaudeCode_描述事实验收与收口.md` C 节：

  · 在看输出之前，用**公开标签 + seed3407** 从 val84 每类固定取 2 图（共 18），
    并加**完整 dev18**，去重后最多 36 图。属**开发审阅**，不叫新盲测/专家考卷。
  · 对 Base、L3072-3407、D3072-3407 的**已有**七字段原答生成**匿名配对**页面：
    原图、morphology、radial_zone、clock_direction、caption_zh。
    **模型名称与公开类别另表**，审阅时不作为依据。
  · 审阅表分列：主要形态是否有图像支持 / 位置是否吻合 / 方向是否适用、吻合 /
    主要结构是否遗漏 / 是否无依据断言 / 格式约定问题。
  · **真实人工栏留空**；若用模型复核，单列模型/上下文/范围，**不能填人工 gold**。
  · 没有可靠字段依据仍写未测；格式与类别不是描述事实评价的替代。
  · 本步骤**不新增 GPU、不重新生成已有模型回答**。

跑法：python 生成审阅集.py
"""
from __future__ import annotations
import hashlib, io, json, random, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
NIGHT = ROOT / "夜间扩容_20261008_0244" / "结果"
IMG_SRC = REPO / "data" / "images"
OUT = HERE / "C_描述事实审阅"
SEED = 3407
MODELS = ["Base", "L-N3072-3407", "D-N3072-3407"]
LABELS = ["甲", "乙", "丙"]

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
FIELDS = ["morphology", "radial_zone", "clock_direction", "caption_zh"]


def load(fname):
    p = NIGHT / fname
    return {r["sample_id"]: r for r in
            (json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip())}


def pick_val84_2_per_class(val_rows):
    """用公开标签 + seed3407 每类固定取 2 图。"""
    by = {}
    for r in val_rows.values():
        if r.get("part") == "val90":
            by.setdefault(r["label"], []).append(r["sample_id"])
    rnd = random.Random(SEED)
    picked = []
    for c in CLASSES:
        pool = sorted(by.get(c, []))
        rnd.shuffle(pool)
        take = pool[:2]
        picked += [(c, s) for s in take]
        print(f"  {c:<10} 候选 {len(pool):>2}  取 {len(take)}  {take}")
    return picked


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "图").mkdir(exist_ok=True)

    data = {m: load(f"{m}_7f_raw.jsonl") for m in MODELS}
    ref = data["L-N3072-3407"]

    print("== 选取：val84 每类 2 图（seed 3407）==")
    picked = pick_val84_2_per_class(ref)
    dev18 = sorted(r["sample_id"] for r in ref.values() if r.get("part") == "dev18")
    print(f"== dev18 全部 {len(dev18)} 图 ==")

    seen, samples = set(), []
    for c, s in picked:
        if s not in seen:
            seen.add(s)
            samples.append({"sample_id": s, "来源": "val84每类2图", "公开类别": c,
                            "part": "val90"})
    for s in dev18:
        if s not in seen:
            seen.add(s)
            samples.append({"sample_id": s, "来源": "完整dev18", "公开类别": ref[s]["label"],
                            "part": "dev18"})
    print(f"== 去重后 {len(samples)} 图 ==")

    # 匿名配对：每图用 sample_id 派生的种子打乱三个模型的顺序
    blocks, key = [], []
    for i, smp in enumerate(samples, 1):
        sid = smp["sample_id"]
        rnd = random.Random(f"{SEED}:{sid}")
        order = MODELS[:]
        rnd.shuffle(order)
        src = IMG_SRC / f"{sid}.png"
        dst = OUT / "图" / f"{sid}.png"
        shutil.copyfile(src, dst)
        sets = []
        for lab, m in zip(LABELS, order):
            r = data[m].get(sid)
            obj = (r or {}).get("obj") or {}
            sets.append({"盲号": lab, "模型": m,
                         "morphology": obj.get("morphology"),
                         "radial_zone": obj.get("radial_zone"),
                         "clock_direction": obj.get("clock_direction"),
                         "caption_zh": obj.get("caption_zh"),
                         "JSON可解析": (r or {}).get("how") == "ok"})
        blocks.append({"图号": i, "sample_id": sid, "来源": smp["来源"], "答案集": sets})
        key.append({"图号": i, "sample_id": sid, "来源": smp["来源"],
                    "公开类别": smp["公开类别"], "part": smp["part"],
                    "盲号对照": {lab: m for lab, m in zip(LABELS, order)},
                    "图sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
        print(f"  图{i:>2} {sid} {smp['公开类别']:<10} 顺序 {order}")

    # ---------- 匿名审阅页（不含模型名、不含公开类别） ----------
    h = ["<!doctype html><html lang='zh'><meta charset='utf-8'>",
         "<title>描述事实审阅集（匿名）</title>",
         "<style>body{font-family:system-ui,sans-serif;max-width:1180px;margin:24px auto;"
         "line-height:1.6}img{width:300px;image-rendering:pixelated;border:1px solid #ccc;"
         "float:left;margin-right:18px}.blk{clear:both;border-top:2px solid #333;"
         "padding:16px 0}.sets{display:flex;gap:14px;flex-wrap:wrap}"
         ".s{flex:1;min-width:260px;background:#f7f7f7;border:1px solid #ddd;"
         "border-radius:6px;padding:10px}.s h4{margin:0 0 6px}.f{color:#555;font-size:12px}"
         ".na{color:#b00}</style>",
         "<h1>描述事实审阅集（匿名配对）</h1>",
         "<p><b>本页不显示模型名称、不显示公开类别</b>——它们的对照留在"
         "<code>盲号对照_另表.md</code>，<b>审阅时不得作为依据</b>。</p>",
         "<p>每张图下面有三个答案集（甲/乙/丙），**顺序已按图打乱**。"
         "请就每个答案集分别回答六个问题：</p>",
         "<ol><li>主要形态是否有图像支持？</li><li>位置是否吻合？</li>"
         "<li>方向是否适用／吻合？（不适用时写「不适用」）</li>"
         "<li>主要结构是否有遗漏？</li><li>是否有无依据的断言？</li>"
         "<li>格式约定问题？（如 <code>clock_direction</code> 写成字符串 "
         "<code>\"null\"</code> 而非 JSON null）</li></ol>",
         "<p>案例属<b>开发审阅</b>，不是新盲测，也不是专家考卷。</p>"]
    for b in blocks:
        h.append(f"<div class='blk'><h3>图 {b['图号']}　<code>{b['sample_id']}</code>"
                 f"　<span class='f'>来源 {b['来源']}</span></h3>")
        h.append(f"<img src='图/{b['sample_id']}.png' alt='{b['sample_id']}'>")
        h.append("<div class='sets'>")
        for s in b["答案集"]:
            g = lambda v: (f"<span class='na'>{v}</span>" if v is None
                           else (str(v) if str(v) else "<span class='na'>（空）</span>"))
            h.append(f"<div class='s'><h4>答案集 {s['盲号']}</h4>"
                     f"<div class='f'>morphology</div><div>{g(s['morphology'])}</div>"
                     f"<div class='f'>radial_zone</div><div>{g(s['radial_zone'])}</div>"
                     f"<div class='f'>clock_direction</div><div>{g(s['clock_direction'])}</div>"
                     f"<div class='f'>caption_zh</div><div>{g(s['caption_zh'])}</div></div>")
        h.append("</div></div>")
    h.append("</html>")
    io.open(OUT / "审阅页.html", "w", encoding="utf-8", newline="\n").write("\n".join(h))

    # ---------- 审阅表（人工栏留空） ----------
    COLS = ["图号", "答案集", "主要形态是否有图像支持", "位置是否吻合",
            "方向是否适用/吻合", "主要结构是否遗漏", "是否无依据断言", "格式约定问题",
            "真实人工复核者", "真实人工结论", "模型复核者", "模型复核上下文/范围",
            "模型复核结论"]
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for b in blocks:
        for s in b["答案集"]:
            lines.append(f"| {b['图号']} | {s['盲号']} |  |  |  |  |  |  |  |  |  |  |  |")
    io.open(OUT / "审阅表_待填.md", "w", encoding="utf-8", newline="\n").write(
        "# 描述事实审阅表（**全部结论栏留空，等人填**）\n\n"
        "> **本表没有任何人工结论。**「真实人工复核者 / 真实人工结论」两列**必须由人填**，"
        "AI 不得代填、不得伪造。\n"
        ">\n"
        "> 「模型复核」三列若填，必须写明**模型名 / 上下文 / 作用范围**，"
        "且**不能冒充人工 gold**，也不能当作唯一测试真值。\n"
        ">\n"
        "> 格式与类别**不是**描述事实评价的替代；没有可靠字段依据的仍写「未测」。\n\n"
        + "\n".join(lines) + "\n")

    # ---------- 盲号对照另表 ----------
    kl = ["# 盲号对照表（**审阅时不得查看**）", "",
          "> 这份表把「图 N 的甲/乙/丙」映射回模型名，并给出公开类别。",
          "> 目的是让审阅在不知道模型、不知道类别的前提下进行；",
          "> **填审阅表之前不要打开这份文件。**", ""]
    for k in key:
        kl.append(f"## 图 {k['图号']}　`{k['sample_id']}`　来源 {k['来源']}　"
                  f"公开类别 **{k['公开类别']}**")
        kl.append("")
        for lab, m in k["盲号对照"].items():
            kl.append(f"- {lab} = `{m}`")
        kl.append("")
    io.open(OUT / "盲号对照_另表.md", "w", encoding="utf-8", newline="\n").write("\n".join(kl))

    # ---------- 选取说明 ----------
    io.open(OUT / "选取说明.md", "w", encoding="utf-8", newline="\n").write(f"""\
# C 段固定描述事实审阅集：怎么选出来的

**性质：开发审阅集。不是新盲测，不是专家考卷，不代表训练/评测划分的任何变化。**
（否则它就变成又一个被优化的目标了。）

## 选取规则（在**看输出之前**就已固定）

1. 从 val84（原答中 `part=="val90"` 的 84 条）里，按**公开标签**每类取 2 图，
   每类内部用 **seed {SEED}** 打乱后取前 2 —— 共 {len(picked)} 图。
2. 加上**完整 dev18**（{len(dev18)} 图）。
3. 两边 `sample_id` 去重 —— 最终 **{len(samples)} 图**。

val84 每类候选数：{'、'.join(f'{c}={sum(1 for r in ref.values() if r.get("part")=="val90" and r["label"]==c)}' for c in CLASSES)}。
`Near_full` 在 val84 里只有 4 条，取 2 条，**这是该类的全部候选的一半，不是全类**。

## 匿名化怎么做

每张图的甲/乙/丙顺序，用 **`seed:{sid}`** 派生随机数**逐图打乱**，
不看模型、不看答案内容。因此同一模型在不同图里可能是甲、也可能是丙 ——
**跨图统计「甲是不是更好」没有意义**，只能逐图比较。

- 审阅页 `审阅页.html`：**只有图 + 三个答案集 + 盲号**，
  不含模型名、不含公开类别。
- 模型名与公开类别在 `盲号对照_另表.md`，**填完审阅表之前不要打开**。

## 六个审阅维度（逐答案集回答）

主要形态是否有图像支持 / 位置是否吻合 / 方向是否适用、吻合 /
主要结构是否遗漏 / 是否无依据断言 / 格式约定问题。

## 明确不做的

- **不新增 GPU、不重新生成已有模型回答** —— 三个模型的七字段原答直接取自夜间那批。
- 没有独立 gold 的字段（morphology / radial_zone / clock_direction / caption_zh 的
  **事实正确性**）**仍然写「未测」**；本审阅集给出的是**逐案例的图像支持性判断**，
  不等于全域事实准确率。
- 类别正确与否**不是**本审阅集的评价对象（那是另一套题面的事）。
""")

    io.open(OUT / "样本清单.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"性质": "开发审阅集（非盲测、非专家考卷）",
                    "选取": {"val84每类2图": len(picked), "dev18": len(dev18),
                             "去重后": len(samples), "seed": SEED},
                    "盲号": LABELS, "模型": MODELS,
                    "注意": "盲号顺序逐图打乱；跨图不可比",
                    "样本": key}, ensure_ascii=False, indent=2) + "\n")

    print(f"\n写出到 {OUT}")
    for f in ["审阅页.html", "审阅表_待填.md", "盲号对照_另表.md", "选取说明.md", "样本清单.json"]:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
