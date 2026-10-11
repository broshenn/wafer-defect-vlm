#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 0 步：准备。
输入：合同 00_总纲与合同.md
输出：prompt_v5.txt（合同 3.1 题面逐字）、atomic_templates.json（合同 5.3 六模板逐字）
做法：从合同 markdown 里程序化提取代码块与表格，避免手抄错字。提取后打印 SHA。
验收：prompt_v5.txt 以"这是公开晶圆 BIN 图"开头、以"确定时为空字符串。"结尾；
      atomic_templates.json 含 T1..T6 六个键。
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "00_合同副本" / "00_总纲与合同.md"


def main():
    text = CONTRACT.read_text(encoding="utf-8")

    # ── 3.1 主题面：取 "### 3.1" 之后第一个围栏代码块 ──
    sec31 = text.split("### 3.1 主题面")[1]
    m = re.search(r"```\n(.*?)\n```", sec31, re.S)
    prompt = m.group(1)
    (HERE / "prompt_v5.txt").write_text(prompt, encoding="utf-8", newline="\n")

    # ── 5.3 原子模板：取表格行 | T? | `...` | ... ──
    sec53 = text.split("### 5.3 原子感知问答模板")[1].split("### 5.4")[0]
    templates = {}
    for row in sec53.splitlines():
        mm = re.match(r"\|\s*(T\d)\s*\|\s*`(.*)`\s*\|", row)
        if mm:
            templates[mm.group(1)] = mm.group(2)
    assert sorted(templates) == ["T1", "T2", "T3", "T4", "T5", "T6"], sorted(templates)
    (HERE / "atomic_templates.json").write_text(
        json.dumps(templates, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    # ── 验收打印 ──
    lines = prompt.split("\n")
    assert lines[0].startswith("这是公开晶圆 BIN 图"), lines[0]
    assert lines[-1].endswith("确定时为空字符串。"), lines[-1]
    for name, p in [("prompt_v5.txt", HERE / "prompt_v5.txt"),
                    ("atomic_templates.json", HERE / "atomic_templates.json")]:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: SHA256={sha} bytes={p.stat().st_size}")
    print("prompt 行数:", len(lines))
    print("模板数:", len(templates))
    print("T6 =", templates["T6"][-40:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
