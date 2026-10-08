#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析器**修正版 v2** + 回归：错误分类纠正，主分**必须不变**。

Codex 指出（《验收与裁决》「另外三处措辞及实现修正」第 3 条）：
`复算_更正口径.py` 里 `except ValueError` 排在 `except json.JSONDecodeError` **之前**。

**为什么这是 bug**：`json.JSONDecodeError` 是 `ValueError` 的**子类**，
Python 按书写顺序匹配，所以后面的分支**永远不可达** ——
「截断」会被误判成「重复键」。这正是我在 D 段解析器里踩过、已修的同一个坑，
但在 `复算_更正口径.py` 里**没回头修**。

**为什么主分不受影响**：两条分支都返回 `(None, 原因)`，
样本**一样被拒**，只是**拒的理由名字**不同。所以：
  · `JSON可解析率`（判据是 `how == "ok"`）**不变**
  · val84 的 Accuracy / Macro-F1（判据是 `_pred`）**不变**
  · **只有「解析判定分布」里 truncated / duplicate_key 两个桶的归属会变**

本脚本把这三件事**实测出来**，而不是在报告里声称修好了。

跑法：python 解析器修正版与回归.py
"""
from __future__ import annotations
import io, json, re, sys
from collections import Counter
from pathlib import Path

# 原答带空的 `<think></think>` 前缀，必须先剥掉 —— 漏了这一步会把**全部**
# 样本误判成解析失败（本脚本第一版就踩了这个坑，靠「102 条全失败」这个
# 明显不可能的结果自查发现的）。
EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)

HERE = Path(__file__).resolve().parent          # …/描述事实验收_20261008_1911/收口_v2
ROOT = HERE.parent                              # …/描述事实验收_20261008_1911
NIGHT = ROOT.parent / "夜间扩容_20261008_0244" / "结果"
sys.path.insert(0, str(ROOT))

TAGS = ["L-N3072-3407", "D-N3072-3407", "D-N3072-3408", "D-N3072-3409", "Base"]
KINDS = ["_raw.jsonl", "_7f_raw.jsonl"]


class NonFinite(ValueError):
    pass


def _decode(text, decodeerror_first):
    """decodeerror_first=True 是**修正版**；False 复刻原版的错误顺序。"""
    if not isinstance(text, str):
        return None, "not_str"
    s = EMPTY_THINK.sub("", text.strip())
    if not s:
        return None, "empty"

    def _nodup(pairs):
        k = [p[0] for p in pairs]
        if len(k) != len(set(k)):
            raise ValueError("dup")
        return dict(pairs)

    def _const(x):
        raise NonFinite(x)
    dec = json.JSONDecoder(object_pairs_hook=_nodup, parse_constant=_const)

    def _tail():
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"

    if decodeerror_first:
        try:
            obj, end = dec.raw_decode(s)
        except NonFinite:
            return None, "non_finite"
        except json.JSONDecodeError:          # ← 修正：子类在前
            return _tail()
        except ValueError:
            return None, "duplicate_key"
    else:
        try:
            obj, end = dec.raw_decode(s)
        except NonFinite:
            return None, "non_finite"
        except ValueError:                    # ← 原版：父类在前，吞掉子类
            return None, "duplicate_key"
        except json.JSONDecodeError:          # 永远不可达
            return None, "truncated"
    rest = s[end:].strip()
    if rest:
        return None, ("multiple_objects" if "{" in rest else "trailing_text")
    return (obj, "ok") if isinstance(obj, dict) else (None, "not_object")


# ---------------- 回归用例 ----------------
CASES = [
    ('{"a": 1}',                        "ok"),
    ('{"a": 1',                         "truncated"),
    ('{"a": 1, "a": 2}',                "duplicate_key"),
    ('{"a": NaN}',                      "non_finite"),
    ('{"a": 1} {"b": 2}',               "multiple_objects"),
    ('{"a": 1} 说明',                    "trailing_text"),
    ('',                                "empty"),
    ('[1,2]',                           "not_object"),
]


def main() -> int:
    print("=" * 78)
    print("一、回归用例：错误分类必须被纠正")
    print("=" * 78)
    bad = 0
    for txt, exp in CASES:
        got_v = _decode(txt, True)[1]
        old_v = _decode(txt, False)[1]
        flag = "✓" if got_v == exp else "✗"
        if got_v != exp:
            bad += 1
        mark = "" if got_v == old_v else f"   （原版误判为 {old_v}）"
        print(f"  {flag} {txt[:32]!r:<36} 修正版={got_v:<16}{mark}")
    print(f"  回归：{len(CASES)-bad}/{len(CASES)} 通过")

    print()
    print("=" * 78)
    print("二、对**全部真实原答**重跑两种口径，看主分是否被改动")
    print("=" * 78)
    print(f"  {'模型':<16}{'题面':<8}{'可解析(旧/新)':>16}{'pred不同':>10}{'判定分布变化':>16}")
    total_diff_pred = 0
    for tag in TAGS:
        for kind in KINDS:
            p = NIGHT / f"{tag}{kind}"
            if not p.exists():
                continue
            rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
            old = Counter(); new = Counter(); diff_pred = 0
            for r in rows:
                raw = r.get("raw")
                ov, oh = _decode(raw, False)
                nv, nh = _decode(raw, True)
                old[oh] += 1; new[nh] += 1
                # pred 定义与原脚本一致：只有 how=="ok" 且 defect_class 合法才算
                op = ov.get("defect_class") if (oh == "ok" and isinstance(ov, dict)) else None
                np_ = nv.get("defect_class") if (nh == "ok" and isinstance(nv, dict)) else None
                if op != np_:
                    diff_pred += 1
            total_diff_pred += diff_pred
            chg = {k: (old.get(k, 0), new.get(k, 0))
                   for k in set(old) | set(new) if old.get(k, 0) != new.get(k, 0)}
            ok_o, ok_n = old.get("ok", 0), new.get("ok", 0)
            print(f"  {tag:<16}{kind.replace('_raw.jsonl','').replace('_7f','七字段') or '类别':<8}"
                  f"{f'{ok_o}/{ok_n}':>16}{diff_pred:>10}"
                  f"{(str(chg) if chg else '无'):>16}")

    print()
    print("=" * 78)
    print("三、结论")
    print("=" * 78)
    print(f"  · **预测不同的样本数：{total_diff_pred}**")
    if total_diff_pred == 0:
        print("  · ✓ 主分（Accuracy / Macro-F1 / 可解析率）**逐条不变** —— "
              "错误分类只影响「拒的理由叫什么」，不影响「拒还是收」")
    else:
        print("  · **有样本的预测发生了变化 —— 必须排查，不能当作纯改名**")
    print("  · 换言之：这一条修的是**可读性**，不是**正确性**；")
    print("    原报告的任何数字都不因它改口 —— 但**理由桶的归属要按新版写**。")
    return 1 if (bad or total_diff_pred) else 0


if __name__ == "__main__":
    sys.exit(main())
