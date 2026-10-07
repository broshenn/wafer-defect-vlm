#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线回归：JSON 可解析判定 与 七字段 schema 判定必须分开、且各自正确。

Codex 2026-10-07 §3 点名要覆盖：空对象、单字段、重复键、坏类型、合法完整对象。
另补：字符串 "null" 不得当 JSON null、bool 不得当类别、缺字段、多字段。
**不发送任何模型请求。**
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_desc_v2 import parse_json, schema7, FIELDS   # noqa: E402

GOOD_JSON = ('{"defect_class":"Center","morphology":"中央致密红簇","radial_zone":"center",'
             '"clock_direction":null,"extent_r":null,"caption_zh":"中心有失效簇",'
             '"uncertainty":"边界不清"}')
GOOD = '<think>\n\n</think>\n\n' + GOOD_JSON

CASES = [
    # (名称, 原始回答, 期望可解析, 期望七字段合法, 期望问题含)
    ("合法完整对象（带空think前缀）", GOOD, True, True, None),
    ("合法完整对象（无前缀）", GOOD_JSON, True, True, None),
    ("仅有 </think> 残留", "</think>\n\n" + GOOD_JSON, False, False, "JSON未解析"),
    ("空对象", "{}", True, False, "缺字段"),
    ("仅 defect_class", '{"defect_class":"Center"}', True, False, "缺字段"),
    ("重复键", '{"defect_class":"Center","defect_class":"Donut",'
              '"morphology":"x","radial_zone":"edge","clock_direction":null,'
              '"extent_r":null,"caption_zh":"y","uncertainty":""}', False, False, "JSON未解析"),
    ("extent_r 是字符串 null", GOOD.replace('"extent_r":null', '"extent_r":"null"'),
     True, False, "extent_r非null"),
    ("defect_class 是 bool", GOOD.replace('"Center"', 'true', 1), True, False, "defect_class非字符串"),
    ("defect_class 越界", GOOD.replace('"Center"', '"Foo"', 1), True, False, "defect_class越界"),
    ("radial_zone 为 null", GOOD.replace('"center"', 'null', 1), True, False, "radial_zone非字符串"),
    ("radial_zone 越界", GOOD.replace('"center"', '"top"', 1), True, False, "radial_zone越界"),
    ("clock_direction 是数字", GOOD.replace('"clock_direction":null', '"clock_direction":7'),
     True, False, "clock_direction类型错"),
    ("morphology 是数字", GOOD.replace('"中央致密红簇"', '42', 1), True, False, "morphology非字符串"),
    ("多字段", GOOD[:-1] + ',"extra":1}', True, False, "多字段"),
    ("截断", GOOD[:-20], False, False, "JSON未解析"),
    ("尾随文本", GOOD + " 谢谢", False, False, "JSON未解析"),
    ("两个对象", GOOD[:-1] + "}{}", False, False, "JSON未解析"),
    ("空字符串", "", False, False, "JSON未解析"),
    ("非 JSON 散文", "晶圆中央有一团红簇。", False, False, "JSON未解析"),
]

res = []
for name, raw, want_parse, want_schema, want_prob in CASES:
    obj, how = parse_json(raw)
    p_ok = (how == "ok")
    s_ok, probs = (schema7(obj) if p_ok else (False, ["JSON未解析"]))
    ok = (p_ok == want_parse) and (s_ok == want_schema)
    if want_prob is not None:
        ok = ok and any(want_prob in x for x in probs)
    res.append({"用例": name, "通过": ok, "可解析": p_ok, "期望可解析": want_parse,
                "七字段合法": s_ok, "期望合法": want_schema,
                "判定码": how, "问题": probs})
    print(f"  {'OK  ' if ok else 'FAIL'} {name:28s} 可解析={p_ok!s:5s}(期望{want_parse!s:5s}) "
          f"合法={s_ok!s:5s}(期望{want_schema!s:5s})  {probs[:1]}")

# 关键分界：v1 会把前三个「可解析但不合法」的用例计成七字段合法率 1.0
sep = [r for r in res if r["可解析"] and not r["七字段合法"]]
print(f"\n可解析但七字段不合法（必须与可解析率分开计）的用例：{len(sep)} 条")
for r in sep:
    print(f"  - {r['用例']}")

fails = [r for r in res if not r["通过"]]
print(f"\n通过 {len(res)-len(fails)}/{len(res)}")
out = Path(__file__).resolve().parent / "回归_解析与schema.json"
io.open(out, "w", encoding="utf-8", newline="\n").write(
    json.dumps({"通过": f"{len(res)-len(fails)}/{len(res)}", "用例": res,
                "note": "离线回归，未发任何模型请求。"},
               ensure_ascii=False, indent=2) + "\n")
print(f"写出 {out.name}")
sys.exit(0 if not fails else 1)
