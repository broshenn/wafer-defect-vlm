#!/usr/bin/env python3
"""单原答机械校验；不修改原文，不判断视觉事实，不读取真值。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]
CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full",
           "Random", "Scratch", "none", "unknown"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]
EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
PROMPT_SHA256 = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"


class DuplicateKey(ValueError):
    pass


class NonFinite(ValueError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def nodup(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise DuplicateKey("duplicate_key")
    return dict(pairs)


def no_constant(value):
    raise NonFinite(value)


def parse_strict(text):
    if not isinstance(text, str):
        return None, "not_str"
    s = EMPTY_THINK.sub("", text.strip())
    if not s:
        return None, "empty"
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=nodup,
                                    parse_constant=no_constant).raw_decode(s)
    except DuplicateKey:
        return None, "duplicate_key"
    except NonFinite:
        return None, "non_finite"
    except json.JSONDecodeError:
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        return None, "not_json"
    rest = s[end:].strip()
    if rest:
        return None, "multiple_objects" if "{" in rest else "trailing_text"
    return (obj, "ok") if isinstance(obj, dict) else (None, "not_object")


def schema7(obj):
    """保持冻结 schema：clock 字符串 null 合法，extent_r 非 null 非法。"""
    if not isinstance(obj, dict):
        return False, ["未解析"]
    problems = []
    missing = [f for f in FIELDS if f not in obj]
    extra = [k for k in obj if k not in FIELDS]
    if missing:
        problems.append("缺字段:" + ",".join(missing))
    if extra:
        problems.append("多字段:" + ",".join(sorted(extra)))
    for key, domain in (("defect_class", CLASSES), ("radial_zone", ZONES)):
        if key in obj:
            if not isinstance(obj[key], str):
                problems.append(key + "非字符串")
            elif obj[key] not in domain:
                problems.append(key + "越界")
    if "clock_direction" in obj:
        value = obj["clock_direction"]
        if not (value is None or isinstance(value, str)):
            problems.append("clock_direction类型错")
    if "extent_r" in obj and obj["extent_r"] is not None:
        problems.append("extent_r非null")
    for key in ("morphology", "caption_zh", "uncertainty"):
        if key in obj and not isinstance(obj[key], str):
            problems.append(key + "非字符串")
    return not problems, problems


def check_answer(answer):
    """可直接传 dict；dict 无逐字原文，不伪造 raw_sha256。"""
    is_dict = isinstance(answer, dict)
    obj, status = (answer, "dict_input") if is_dict else parse_strict(answer)
    parsed = status in ("ok", "dict_input")
    schema_ok, problems = schema7(obj)
    warnings = []
    if is_dict:
        warnings.append("dict_input:无法从对象恢复逐字原文或原始重复键")
    if obj is not None and obj.get("clock_direction") == "null":
        warnings.append("clock_direction_string_null:类型合法，但违反无可靠方向时填JSON null的约定")
    return {"strict_json": parsed, "parse_status": status,
            "schema_ok": bool(parsed and schema_ok), "schema_problems": problems,
            "contract_warnings": warnings,
            "raw_text": answer if isinstance(answer, str) else None,
            "raw_sha256": sha(answer.encode("utf-8")) if isinstance(answer, str) else None,
            "canonical_object_sha256": sha(json.dumps(answer, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8")) if is_dict else None,
            "validator_sha256": sha(Path(__file__).read_bytes()),
            "scope": "仅结构/类型/域；不等于视觉事实正确或人工gold"}


def self_test():
    base = {"defect_class": "unknown", "morphology": "离线合成检查文本",
            "radial_zone": "unknown", "clock_direction": None, "extent_r": None,
            "caption_zh": "离线合成检查文本", "uncertainty": ""}
    dump = lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    valid = dump(base)
    missing = dict(base); missing.pop("uncertainty")
    wrong_types = {**base, "defect_class": [], "radial_zone": 1,
                   "clock_direction": 3, "morphology": None,
                   "caption_zh": {}, "uncertainty": False}
    cases = [
        ("合法七字段", valid, True, True, False),
        ("允许空think前缀", "<think>\n\n</think>\n\n" + valid, True, True, False),
        ("clock字符串null主schema仍通过", dump({**base, "clock_direction": "null"}), True, True, True),
        ("重复键", valid[:-1] + ',"uncertainty":""}', False, False, False),
        ("缺字段", dump(missing), True, False, False),
        ("多字段", dump({**base, "extra": "合成字段"}), True, False, False),
        ("类别别名不改写", dump({**base, "defect_class": "Edge-Ring"}), True, False, False),
        ("径向中文非枚举", dump({**base, "radial_zone": "边缘"}), True, False, False),
        ("字段类型错误", dump(wrong_types), True, False, False),
        ("extent字符串null不通过", dump({**base, "extent_r": "null"}), True, False, False),
        ("围栏原答失败", "```json\n" + valid + "\n```", False, False, False),
        ("多对象尾随", valid + "\n" + valid, False, False, False),
        ("非有限常量", valid.replace('"extent_r":null', '"extent_r":NaN'), False, False, False),
    ]
    results = []
    for name, raw, expect_json, expect_schema, expect_warn in cases:
        actual = check_answer(raw)
        ok = (actual["strict_json"] == expect_json and actual["schema_ok"] == expect_schema
              and bool(actual["contract_warnings"]) == expect_warn)
        results.append({"name": name, "passed": ok,
                        "expected": {"strict_json": expect_json, "schema_ok": expect_schema,
                                     "has_contract_warning": expect_warn}, "actual": actual})
    return {"passed": sum(x["passed"] for x in results), "total": len(results),
            "all_passed": all(x["passed"] for x in results), "cases": results,
            "validator_sha256": sha(Path(__file__).read_bytes()),
            "note": "全部为离线合成反例，无真实图片标签或模型答案"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", help="单原答UTF-8文件；-表示标准输入")
    ap.add_argument("--input", dest="input_flag", help="同位置参数input")
    ap.add_argument("--raw-key", choices=("raw_answer", "raw"),
                    help="仅用于JSON记录封装；明确选取原答键，避免猜测")
    ap.add_argument("--out", help="另写校验JSON，不覆盖输入")
    ap.add_argument("--self-test", action="store_true", help="运行13个离线合成检查")
    args = ap.parse_args()
    if args.self_test:
        result = self_test(); exit_code = 0 if result["all_passed"] else 1
    else:
        name = args.input_flag or args.input
        if not name:
            ap.error("需要单原答input或--self-test")
        data = sys.stdin.buffer.read() if name == "-" else Path(name).read_bytes()
        text = data.decode("utf-8-sig")
        answer = text
        if args.raw_key:
            envelope, status = parse_strict(text)
            if status != "ok" or args.raw_key not in envelope:
                ap.error("封装JSON非法、含重复键或未包含指定原答键")
            answer = envelope[args.raw_key]
        result = check_answer(answer)
        result["input_sha256"] = sha(data)
        result["input_path"] = name
        result["input_mode"] = args.raw_key or "verbatim_text"
        prompt_path = Path(__file__).with_name("prompt_7f.txt")
        result["prompt_file_sha256"] = sha(prompt_path.read_bytes())
        result["prompt_hash_ok"] = result["prompt_file_sha256"] == PROMPT_SHA256
        exit_code = 0 if result["schema_ok"] and result["prompt_hash_ok"] else 1
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        dest = Path(args.out)
        input_name = args.input_flag or args.input
        if not args.self_test and input_name != "-" and dest.resolve() == Path(input_name).resolve():
            ap.error("校验输出不得覆盖原答")
        with dest.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(output)
    else:
        print(output, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
