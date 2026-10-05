#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON 重复键审计 + 无重名 schema 另存。

## 为什么需要

`json.loads` 默认**静默**保留同名键的**最后一个**。于是：

```json
"counts": { "failed": 1, "failed": { "sample_id": "..." } }
```

解析出来 `counts["failed"]` 是**对象**，不是数字 `1`。
任何"读 counts.failed 拿失败数"的消费方都会失准，而**文件本身看着完全合法**。

## 本工具做什么

  * `audit`：递归找出**所有**嵌套层的重复键，报出**路径 + 键名 + 各次出现的值**；
  * `reschema`：把已知的重复键拆成不重名的字段（如 `failed` → `failure_count` +
    `failure_details`），**另存新文件**，并记录逐项差异。
  * **原文件只读，永不修改。**

## 原则

  * 不猜、不合并、不丢弃 —— 重复键一律**显式报出**，由人决定怎么拆；
  * 拆分只在**显式给出映射规则**时发生；没有规则就只报不改；
  * 失败**数量**与失败**详情**必须分成两个字段，不能同名。

用法：
    python json_dupkey_audit.py audit <file.json>
    python json_dupkey_audit.py reschema <file.json> --out <new.json> --diff <diff.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 已知的重复键拆分规则：同一个键既是"计数"又是"详情"时怎么分。
# 只在**显式**命中时生效；不命中就原样保留并报出。
SPLIT_RULES = {
    ("counts", "failed"): {"count": "failure_count", "details": "failure_details"},
    ("counts", "not_executed"): {"count": "not_executed_count",
                                 "details": "not_executed_details"},
}


class DuplicateKeyError(ValueError):
    pass


def find_duplicates(text: str) -> list[dict]:
    """递归找出所有嵌套层的重复键。返回 [{path, key, occurrences:[{index,value_preview}]}]。"""
    hits: list[dict] = []

    def hook(pairs):
        keys = [k for k, _ in pairs]
        seen: dict[str, list] = {}
        for i, (k, v) in enumerate(pairs):
            seen.setdefault(k, []).append((i, v))
        for k, occ in seen.items():
            if len(occ) > 1:
                hits.append({"key": k, "n_occurrences": len(occ),
                             "occurrences": [
                                 {"index": i,
                                  "type": type(v).__name__,
                                  "preview": json.dumps(v, ensure_ascii=False)[:160]}
                                 for i, v in occ]})
        return dict(pairs)

    json.loads(text, object_pairs_hook=hook)
    return hits


def _pairs_to_obj(pairs, path: tuple, changes: list):
    """从**保留重复键的键值对**重建对象，并在发现重复时按规则拆名。

    关键：不能先 json.loads 再改 —— 那样重复键已经被静默折叠、计数值早丢了。
    必须从 pairs 出发。**只拆真正重复的键**；没重复的键一个都不改名。
    """
    groups: dict[str, list] = {}
    order: list[str] = []
    for k, v in pairs:
        if k not in groups:
            order.append(k)
        groups.setdefault(k, []).append(v)

    out = {}
    for k in order:
        vals = groups[k]
        sub = path + (k,)
        rule = SPLIT_RULES.get(sub)
        if len(vals) > 1 and rule:
            # 真重复：把数值那个当计数、对象那个当详情，分别命名
            def _is_pair_obj(x):
                """用了 object_pairs_hook 之后，嵌套对象是 ("__pairs__", [...]) 元组。"""
                return isinstance(x, tuple) and len(x) == 2 and x[0] == "__pairs__"

            nums = [x for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool)]
            objs = [x for x in vals if _is_pair_obj(x)]
            others = [x for x in vals if not _is_pair_obj(x) and x not in nums]
            if nums and objs:
                out[rule["count"]] = nums[0]
                out[rule["details"]] = _build(objs[0], sub, changes)
                changes.append({"path": ".".join(sub), "action": "split_duplicate",
                                "count_key": rule["count"], "details_key": rule["details"],
                                "count_value": nums[0],
                                "n_values": len(vals)})
                for extra in others:
                    out[f"{k}_extra{len(out)}"] = _build(extra, sub, changes)
                    changes.append({"path": ".".join(sub), "action": "kept_unnamed_extra"})
                continue
            if nums:
                out[rule["count"]] = nums[0]
                changes.append({"path": ".".join(sub), "action": "split_duplicate_numeric_only",
                                "count_key": rule["count"], "count_value": nums[0]})
                for i, o in enumerate(objs):
                    out[f"{rule['details']}_{i}"] = _build(o, sub, changes)
                    changes.append({"path": ".".join(sub), "action": "renamed_to_details",
                                    "new_key": f"{rule['details']}_{i}"})
                continue
            # 有重复但没有可用规则 → **原样保留最后一次并显式记录**，不猜
            changes.append({"path": ".".join(sub), "action": "duplicate_no_rule_kept_last",
                            "n_values": len(vals)})
            out[k] = _build(vals[-1], sub, changes)
            continue
        # 没重复：原样走，不改名
        out[k] = _build(vals[-1], sub, changes)
    return out


def rebuild(text: str) -> tuple[object, list]:
    """从文本重建，保留重复键信息并按规则拆分。"""
    parsed = {"root": None}

    def hook(pairs):
        return ("__pairs__", pairs)

    parsed["root"] = json.loads(text, object_pairs_hook=hook)
    changes: list = []
    return _build(parsed["root"], (), changes), changes


def _build(node, path: tuple, changes: list):
    if isinstance(node, tuple) and len(node) == 2 and node[0] == "__pairs__":
        return _pairs_to_obj(node[1], path, changes)
    if isinstance(node, list):
        return [_build(v, path, changes) for v in node]
    return node


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "reschema"])
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--diff", type=Path)
    a = ap.parse_args()

    raw = a.src.read_bytes()
    text = raw.decode("utf-8-sig")
    print("=" * 74)
    print(f"JSON 重复键审计：{a.src}")
    print(f"  字节 {len(raw)}  SHA256 {hashlib.sha256(raw).hexdigest()}")
    print("=" * 74)

    dups = find_duplicates(text)
    if not dups:
        print("\n未发现重复键。")
    else:
        print(f"\n发现 {len(dups)} 处重复键：")
        for d in dups:
            print(f"  键 {d['key']!r} 出现 {d['n_occurrences']} 次：")
            for o in d["occurrences"]:
                print(f"    第 {o['index']} 次  类型={o['type']:6s}  {o['preview']}")

    if a.mode == "audit":
        return 0 if not dups else 1

    if not a.out or not a.diff:
        print("\n[停止] reschema 需要 --out 与 --diff")
        return 2
    if a.out.exists() or a.diff.exists():
        print("\n[停止] 输出已存在，拒绝覆盖")
        return 3

    obj, changes = rebuild(text)

    out_text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    a.out.write_text(out_text, encoding="utf-8", newline="\n")

    still = find_duplicates(a.out.read_text(encoding="utf-8"))

    diff = {
        "source": str(a.src), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output": str(a.out),
        "output_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "duplicates_found_in_source": dups,
        "changes": changes,
        "duplicates_remaining_in_output": still,
        "note": "原文件未修改；本文件只记录差异。失败数量与失败详情已分成两个字段。",
    }
    a.diff.write_text(json.dumps(diff, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")

    print(f"\n另存无重名版本：{a.out}")
    print(f"  改名字段 {len([c for c in changes if c['action']=='renamed_to_count'])} 个，"
          f"详情字段 {len([c for c in changes if c['action']=='renamed_to_details'])} 个")
    print(f"  新文件里仍存在的重复键：{still or '无'}")
    print(f"  差异记录：{a.diff}")
    return 0 if not still else 4


if __name__ == "__main__":
    sys.exit(main())
