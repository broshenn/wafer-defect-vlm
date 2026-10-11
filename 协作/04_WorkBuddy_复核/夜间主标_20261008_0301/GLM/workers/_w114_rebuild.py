# -*- coding: utf-8 -*-
"""分片114 重建：glob发现权威清单，按看图判断写临时件。"""
import glob
import json
import os
import sys

BASE = "D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"
TMP = os.path.join(BASE, "_tmp_114.jsonl")

ANSWERS = {
    "wbg0792": ("Random", "global", None,
        "红色失效die以单颗或2-3颗小簇散布于全晶圆，无连续成片、线状或环形结构，边缘与中心均可见散点",
        "失效die随机散布于整个晶圆，未观察到明显聚集或几何图案",
        "边缘与中心密度差异不明显，无法排除轻微边缘偏好"),
    "wbg0793": ("Center", "center", None,
        "晶圆中心区有密集红色失效die团聚，呈不规则团块并向外延伸出数条短枝状红色die链；其余区域为孤立的红色散点",
        "主要形态为中心致密失效团块伴少量外伸短链，周围散布零星失效die",
        "中心团块与周围散点是否同一成因无法从图中判断"),
    "wbg0794": ("Center", "center", "3点钟",
        "红色失效die在中心聚集为不规则团块，并向3点钟方向延伸出一条较长的红色die链，另见数条短枝；周围区域散布零星失效die",
        "中心致密失效团块伴3点钟方向长伸链，其余区域少量孤立失效die",
        "中心团块内夹杂少量绿色die；延伸链与中心团是否同一失效成因无法从图中判断"),
    "wbg0795": ("Center", "center", None,
        "红色失效die在晶圆中心区域密集聚集并连成小块，失效密度由中心向外逐渐降低，边缘区仅有零星孤立失效die",
        "中心密集失效并向外递减的斑驳分布，未见环形或线状结构",
        "中心向外过渡是渐进的，无法明确划定中心区的半径边界"),
    "wbg0796": ("Edge_Loc", "edge", "3-5点钟",
        "红色失效die密集分布于晶圆边缘带，右侧（3点钟至5点钟）最重并连成宽带，其余边缘为断续散点；内部仅有少量孤立失效die",
        "边缘带状密集失效以右侧为最，内部相对干净",
        "图中中心可见1颗黑色die，含义不明；边缘带内夹杂绿色die，带状宽度无法精确判定"),
    "wbg0797": ("Center", "center", None,
        "红色失效die在晶圆中部偏内区域密集聚集，形成不规则团块并向外伸出多条枝状红色die链；外围仅有零星孤立失效die",
        "中部密集失效团块伴多条外伸短链，外围相对干净",
        "团块位置略偏离几何中心，中心区与中部区边界无法精确划定"),
    "wbg0798": ("Loc", "center", None,
        "红色失效die在晶圆中部偏上区域密集连片，形成不规则团块并向12点钟、3点钟、5点钟方向伸出数条短链；外围散布零星失效die",
        "中部偏上不规则团块伴多方向短链，外围散点稀疏",
        "团块偏离几何中心，与单纯中心聚集的区分依据不足"),
}


def main():
    hits = glob.glob("D:/pycode/晶圆图研究/协作/**/WB_gap1200_*114*.jsonl", recursive=True)
    assert len(hits) == 1, "manifest hits=%d" % len(hits)
    rows = [json.loads(l) for l in open(hits[0], encoding="utf-8") if l.strip()]
    assert len(rows) == 7, "manifest rows != 7"
    with open(TMP, "w", encoding="utf-8") as f:
        for r in rows:
            iid = r["item_id"]
            cls, zone, clock, morph, cap, unc = ANSWERS[iid]
            ans = {"defect_class": cls, "morphology": morph, "radial_zone": zone,
                   "clock_direction": clock, "extent_r": None,
                   "caption_zh": cap, "uncertainty": unc}
            rec = {"item_id": iid, "image_sha256": r["image_sha256"],
                   "raw_answer": json.dumps(ans, ensure_ascii=False)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("WROTE", len(rows), "rows ->", TMP)


if __name__ == "__main__":
    main()
