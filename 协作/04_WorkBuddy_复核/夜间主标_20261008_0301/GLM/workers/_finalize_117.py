# -*- coding: utf-8 -*-
"""分片117 定稿脚本：写临时 -> 校验 -> 生成最终文件 -> 自确认"""
import json, hashlib, io, os, sys
from datetime import datetime

BASE = r"D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"
TMP = os.path.join(BASE, "_tmp_117.jsonl")
FINAL = os.path.join(BASE, "分片117_原答.jsonl")
PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
MODEL = "GLM-5.3-Flash"

# 清单（哈希已于第一步用 hashlib 核验 7/7 MATCH，此处照抄清单值）
MANIFEST = [
    ("wbg0813", "b073913356f1a15db7112c083a57b7a4d9d0e0ec0399b1292f935850aa95f875"),
    ("wbg0814", "eceda808880e86edd5a44e074694ff12f9cb4bba6c1b73b3ea3d2ec25e0d65d2"),
    ("wbg0815", "b010931ff489bfe7b67a19e01193950169dd63f5851eb3003249ef2f6f263b8a"),
    ("wbg0816", "32fc2a8675fec44d2e52ac63b3c5af9d5f1d7b241eadb3b0d9c058aae55d5256"),
    ("wbg0817", "d51e46732807637fa8b5c919d899f5b4ae30bafa88831ef6e5d4a7981ccb268d"),
    ("wbg0818", "87b489fd307f6d6ea10fbade5a375d5c8a160353c26b348630d7cc9660a49470"),
    ("wbg0819", "fb6d43ebc0260bb12c36e4265b24bf1d6ba6e04d97468052e6473a802d6f08fb"),
]

ANSWERS = {
    "wbg0813": (
        "1) 缺陷类别：Random（随机散布）。红色失效 die 以孤立单点为主，无中心聚集、无环形带、"
        "无长条划痕，属单一类别。"
        "2) 位置与分布形态：红色小点散布全圆，上部与中带略密，左上（约10-11点钟）稍多，其余零星分布；"
        "未见连续成片的失效区域。"
        "3) 尺寸/覆盖比例：失效点多为单 die 尺寸，估计覆盖面积约3%-4%。"
        "4) 方向性：中部有几处 2 die 水平相邻的小连片（约9-3点钟走向），但长度很短，不构成划痕。"
        "5) 其他：绿色 pass die 占绝对主体，圆片轮廓完整；边缘一圈偶有贴边红点，属正常边缘效应。"
    ),
    "wbg0814": (
        "1) 缺陷类别：Random（随机散布）。"
        "2) 位置与分布形态：红点分布全圆，中部（中心偏左下）密度略高，边缘散点较少；"
        "均为孤立单点或 2 点小连片，无成片失效。"
        "3) 尺寸/覆盖比例：以单点失效为主，估计覆盖约4%-5%。"
        "4) 方向性：无明显方向性；右上边缘有一颗 2 die 水平短连片（约9-3点钟走向）。"
        "5) 其他：绿色主体完好，轮廓完整，无 Edge-Ring/Donut/Center 结构。"
    ),
    "wbg0815": (
        "1) 缺陷类别：none 为主（近正常晶圆），仅极少量 Random 散点。"
        "2) 位置与分布形态：红色失效点极少且分散，右缘（约2-4点钟）有几颗黑色 die 贴边，疑为边缘失效；"
        "左缘中部有一颗 2 die 水平短连片。"
        "3) 尺寸/覆盖比例：覆盖极低，约1.5%-2.5%，均为单 die 尺度。"
        "4) 方向性：无方向性条带。"
        "5) 其他：绿色主体占比约97%以上，无任何聚集核、环带或弧带，整体接近正常晶圆。"
    ),
    "wbg0816": (
        "1) 缺陷类别：Random（随机散布），边缘略密。"
        "2) 位置与分布形态：红点散布全圆，左上至上缘（10-12点钟）与左缘略密，底部边缘有数颗；"
        "多为单点，偶见 2 点相邻。"
        "3) 尺寸/覆盖比例：估计覆盖约3%-5%。"
        "4) 方向性：无明显条带方向；边缘散点大致沿圆周分布。"
        "5) 其他：绿色主体完好，无环带或中心斑；上缘一颗 2 die 水平连片，属短连片而非划痕。"
    ),
    "wbg0817": (
        "1) 缺陷类别：Random（随机散布），伴一处短横条与一处边缘竖条。"
        "2) 位置与分布形态：红点稀疏散布全圆；中心偏上（约12-1点钟内侧）有一颗 2-3 die 水平短连片；"
        "右缘3点钟附近有一列 3-4 die 竖向贴边连片；其余散点随机。"
        "3) 尺寸/覆盖比例：估计覆盖约2%-3%。"
        "4) 方向性：两处小连片分别近水平（9-3点钟）与竖向（12-6点钟），但长度均很短，未达划痕级别。"
        "5) 其他：绿色主体完好，无环带/中心斑；右缘竖条贴近轮廓，可能为边缘 die 连续失效。"
    ),
    "wbg0818": (
        "1) 缺陷类别：Random（随机散布），中右部略密。"
        "2) 位置与分布形态：红点散布全圆，中右（1-5点钟内侧）密度略高，可见 2-4 die 小连片 2-3 处；"
        "底部边缘有数颗贴边红点；左半部较稀。"
        "3) 尺寸/覆盖比例：估计覆盖约4%-6%。"
        "4) 方向性：无明显整体方向性；个别小连片近水平走向。"
        "5) 其他：绿色主体完好，无 Edge-Ring/Donut/Center；失效以小连片+散点混合，仍属随机型。"
    ),
    "wbg0819": (
        "1) 缺陷类别：Random（随机散布）伴局部小簇，混合形态但不构成单一 Loc/Center。"
        "2) 位置与分布形态：红点散布全圆；中心偏右上（约1-2点钟内侧）有一个约 4-6 die 的斜向/弧形小簇；"
        "右下（约4-5点钟内侧）有一列 3-4 die 近竖向点列；其余散点随机。"
        "3) 尺寸/覆盖比例：估计覆盖约4%-6%，最大连片约 4-6 die。"
        "4) 方向性：中心右上小簇呈斜向延展；右下点列近竖向（12-6点钟方向），长约 3-4 die，未达长划痕级别。"
        "5) 其他：绿色主体完好，无闭合环带或实心中心斑；两处小簇值得复核。"
    ),
}

def main():
    # --- ① 写临时文件 ---
    with io.open(TMP, "w", encoding="utf-8") as f:
        for iid, sha in MANIFEST:
            rec = {"item_id": iid, "image_sha256": sha, "raw_answer": ANSWERS[iid]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- ② 校验临时文件 ---
    lines = io.open(TMP, "r", encoding="utf-8").read().splitlines()
    assert len(lines) == 7, "line count != 7: %d" % len(lines)
    seen = []
    parsed = []
    for ln in lines:
        rec = json.loads(ln)
        assert len(rec.get("image_sha256", "")) == 64, "sha len != 64"
        assert rec.get("raw_answer", "").strip() != "", "empty raw_answer"
        seen.append(rec["item_id"])
        parsed.append(rec)
    assert len(set(seen)) == 7, "duplicate item_id"
    assert set(seen) == {m[0] for m in MANIFEST}, "item_id mismatch vs manifest"
    # image_sha256 必须与清单一致
    sha_map = dict(MANIFEST)
    for rec in parsed:
        assert rec["image_sha256"] == sha_map[rec["item_id"]], "sha != manifest"

    # --- ③ 生成最终文件 ---
    now_iso = datetime.now().replace(microsecond=0).isoformat()
    with io.open(FINAL, "w", encoding="utf-8") as f:
        for rec in parsed:
            ans = rec["raw_answer"]
            out = {
                "item_id": rec["item_id"],
                "image_sha256": rec["image_sha256"],
                "prompt_sha256": PROMPT_SHA,
                "raw_answer": ans,
                "answer_sha256": hashlib.sha256(ans.encode("utf-8")).hexdigest(),
                "model": MODEL,
                "answer_time": now_iso,
                "check": "self_ok",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # --- ④ 自确认：最终文件存在于磁盘 ---
    assert os.path.isfile(FINAL), "FINAL not on disk"
    sz = os.path.getsize(FINAL)
    n = sum(1 for _ in io.open(FINAL, "r", encoding="utf-8"))
    print("FINAL_OK", FINAL.replace("\\", "/"), "bytes=%d lines=%d" % (sz, n))
    print("TMP_OK", TMP.replace("\\", "/"))

if __name__ == "__main__":
    main()
