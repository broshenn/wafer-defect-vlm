#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断（**不是成绩**）：统计有多少条作答仅仅因为 `zone_frac` 写成百分数刻度而不合法。

冻结题面只说「比例……各保留一位小数」，没有明确是 0–1 的小数还是 0–100 的百分数；
冻结正则 `(\\d\\.\\d)` 只接受前者。本脚本只做计数，不改变任何正式分数。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("common_v5_pin", HERE / "inputs" / "common_v5.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

PERMITTED = re.compile(
    r"^<measure>red_ratio=(\d\.\d{2}); zone_frac=(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?); "
    r"R1=(\d\.\d); R2=(\d\.\d)</measure>\s*\n\s*(\{.*\})\s*$", re.DOTALL)
ZONE = re.compile(r"zone_frac=[\d.]+,[\d.]+,[\d.]+")


def main() -> int:
    out = {}
    for name in ("BL_maxplus", "BL_small", "BL_flagship"):
        p = HERE / "sources" / f"{name}_raw.jsonl"
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        base = sum(1 for r in rows if c.parse_v5(r.get("raw") or "")["ok"])
        pct_written = rescued = 0
        for r in rows:
            t = r.get("raw") or ""
            m = PERMITTED.match(t)
            if not m:
                continue
            vals = [float(x) for x in m.groups()[1:4]]
            # 真正的百分数刻度签名：整数部分有两位以上（如 33.3），单靠 (\d\.\d) 匹配不上
            if max(vals) < 10.0:
                continue
            pct_written += 1
            if c.parse_v5(t)["ok"]:
                continue
            total = sum(vals) or 1.0
            norm = [v / total for v in vals]
            # 只把 zone_frac 换成归一化到 0–1 的同比例值，其余一律不动
            t2 = ZONE.sub("zone_frac=%.1f,%.1f,%.1f" % tuple(norm), t, count=1)
            if c.parse_v5(t2)["ok"]:
                rescued += 1
        out[name] = {"rows": len(rows), "legal_official": base,
                     "zone_frac_percent_scale": pct_written,
                     "would_be_legal_if_percent_accepted": rescued}
        print(f"{name:14s} 正式合法 {base}/200；写百分数刻度 {pct_written} 条；"
              f"其中仅因刻度而不合法 {rescued} 条")
    (HERE / "诊断_zone_frac刻度.json").write_text(
        json.dumps({"note": "诊断，不是成绩；不改变任何正式分数",
                    "prompt_ambiguity": "题面只说『比例……各保留一位小数』，未明确 0–1 还是 0–100",
                    "results": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
