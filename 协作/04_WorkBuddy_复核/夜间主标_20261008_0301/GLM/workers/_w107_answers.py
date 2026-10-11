# -*- coding: utf-8 -*-
import json, sys

ITEMS = {
    "wbg0743": "503845111f6896e816d9981661d5114813753203959784254217cca5012a6c93",
    "wbg0744": "aa2d7cd054512c8e2de4d0780d3139eba76a24a544b2ddec07e1cacf334d7ffc",
    "wbg0745": "701ce3466bd4482a9c2ca1b27cfd0c87501591874ff7b3726bfe185a3b60e549",
    "wbg0746": "0c7a19f2347c34a76d9ff8318212daf03b14980b863d8e16c3da45ea88420bae",
    "wbg0747": "42cd7f904d4e6ce08ce76ce74c3f2e270b90ad33f3fa789b2c5b91d628a3430a",
    "wbg0748": "0138f1dbbf3d79fee5fa2f948409313f20078aed2de588edabeef45ee2fc663a",
    "wbg0749": "993ef0b070eaa09d103c892633a9cac7fec03af77aab4b86ce1f532921e93c95",
}

ANSWERS = {
    "wbg0743": {
        "defect_class": "Near_full",
        "morphology": "绝大多数 die 呈红色失效，仅零星散布少量绿色合格 die",
        "radial_zone": "global",
        "clock_direction": None,
        "extent_r": None,
        "caption_zh": "全圆片绝大部分 die 失效，仅在上部、左侧与中下部散在少量绿色合格 die",
        "uncertainty": "",
    },
    "wbg0744": {
        "defect_class": "Random",
        "morphology": "红色失效 die 呈孤立单点或小块随机散布于全片，无聚集或线性走向",
        "radial_zone": "global",
        "clock_direction": None,
        "extent_r": None,
        "caption_zh": "全片散布随机分布的红色失效 die，其余大部分为绿色合格 die",
        "uncertainty": "",
    },
    "wbg0745": {
        "defect_class": "Random",
        "morphology": "绿色合格 die 为主，红色失效 die 以孤立点散在分布，下半部略多但无成片或条带",
        "radial_zone": "global",
        "clock_direction": None,
        "extent_r": None,
        "caption_zh": "红色失效 die 散在分布于全片，中上部偏少、下部略密，未见明显聚集或线性图案",
        "uncertainty": "上下的疏密差异较弱，不排除随机波动",
    },
    "wbg0746": {
        "defect_class": "Loc",
        "morphology": "全片散布零星红色失效 die，底部 6 点钟方向边缘处有一片较集中的红色聚集区",
        "radial_zone": "edge",
        "clock_direction": "6点钟",
        "extent_r": None,
        "caption_zh": "底缘 6 点钟方向可见红色失效 die 局部聚集，其余区域为散点分布的失效 die",
        "uncertainty": "聚集区与随机散点并存，聚集范围判断受图像分辨率限制",
    },
    "wbg0747": {
        "defect_class": "Loc",
        "morphology": "红色失效 die 多为散点，右上方边缘处可见一小片相对集中的红色块",
        "radial_zone": "edge",
        "clock_direction": "2点钟",
        "extent_r": None,
        "caption_zh": "右上方边缘约 2 点钟方向有红色失效 die 小范围聚集，其余为全片散点",
        "uncertainty": "聚集范围较小，与随机散点的界限不完全清晰",
    },
    "wbg0748": {
        "defect_class": "Center",
        "morphology": "圆心附近有一片不规则形状的红色失效 die 聚集，周边另有全片散点",
        "radial_zone": "center",
        "clock_direction": None,
        "extent_r": None,
        "caption_zh": "中央区域可见红色失效 die 聚集成团，外围散布零星失效 die",
        "uncertainty": "中央聚集形状不规则且略偏离圆心，边界不清晰",
    },
    "wbg0749": {
        "defect_class": "Scratch",
        "morphology": "顶部有一条横向连续的红色条带，另有全片零散红色失效点",
        "radial_zone": "edge",
        "clock_direction": "12点钟",
        "extent_r": None,
        "caption_zh": "顶部 12 点钟方向附近可见一条水平走向的红色失效条带，其余区域散布零星失效 die",
        "uncertainty": "条带起点与终点被边缘截断，实际延伸范围不明",
    },
}

def append_answer(item_id, ans):
    line = {
        "item_id": item_id,
        "image_sha256": ITEMS[item_id],
        "raw_answer": json.dumps(ans, ensure_ascii=False),
    }
    path = "协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers/_tmp_107.json"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print("appended", item_id)

if __name__ == "__main__":
    item_id = sys.argv[1]
    append_answer(item_id, ANSWERS[item_id])
