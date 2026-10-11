#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mechanical read-only audit; no model, image inference, network, or GPU use.

Every normal run writes a new snapshot below this script's directory.
Saved paths are repository-relative. Worker files are never modified.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

WORKERS = "协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"
INDEX = ("协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/"
         "WorkBuddy_训练缺口1200/WB_gap1200_索引.jsonl")
PROMPT = "协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/锁定题面_结构化_含不确定.txt"
PROTOCOL = "协作/02_ClaudeCode_实操/同图ABC准备_20261007/描述评测协议.json"
SCHEMA_SOURCE = ("协作/02_ClaudeCode_实操/描述SFT与同题评测_20261007_1430/"
                 "修正_v2/eval_desc_v2.py")
EXPECTED_PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
FIELDS = ("defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty")
CLASSES = ("Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none", "unknown")
ZONES = ("center", "middle", "edge", "global", "none", "unknown")
CANONICAL = re.compile(r"分片\d{3}_原答\.jsonl")
TZ = dt.timezone(dt.timedelta(hours=8))


class DuplicateKey(ValueError):
    pass


class NonFinite(ValueError):
    pass


def no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateKey(key)
        obj[key] = value
    return obj


def no_nonfinite(value):
    raise NonFinite(value)


def strict_loads(text):
    return json.loads(text, object_pairs_hook=no_duplicates,
                      parse_constant=no_nonfinite)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def find_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    raise RuntimeError("Cannot locate repository root via AGENTS.md")


def relative(path, root):
    return path.relative_to(root).as_posix()


def now():
    return dt.datetime.now(TZ).isoformat(timespec="seconds")


def file_snapshot(path, root):
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    return content, {
        "path": relative(path, root), "bytes_read": len(content),
        "sha256_read": sha(content),
        "mtime": dt.datetime.fromtimestamp(after.st_mtime, TZ).isoformat(timespec="seconds"),
        "mtime_ns": after.st_mtime_ns,
        "changed_during_read": (before.st_size, before.st_mtime_ns) !=
                               (after.st_size, after.st_mtime_ns),
    }


def saved_jsonl(content):
    rows, problems = [], []
    for number, line in enumerate(content.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        if not line.endswith(b"\n"):
            problems.append({"line": number, "code": "unterminated_tail_not_counted"})
            continue
        try:
            obj = strict_loads(line.decode("utf-8"))
            if not isinstance(obj, dict):
                raise TypeError("record_not_object")
        except (ValueError, TypeError, UnicodeError) as exc:
            problems.append({"line": number, "code": type(exc).__name__})
            continue
        rows.append((number, obj))
    return rows, problems


def parse_answer(raw):
    """Strict whole-value parse; no wrapper or substring repair."""
    if isinstance(raw, dict):
        return raw, "dict_object"
    if not isinstance(raw, str):
        return None, "unsupported_raw_type"
    if not raw.strip():
        return None, "empty_string"
    try:
        obj = strict_loads(raw.strip())
    except DuplicateKey:
        return None, "duplicate_key"
    except NonFinite:
        return None, "non_finite"
    except json.JSONDecodeError:
        shaped = raw.lstrip().startswith(("{", "[", '"', chr(96) * 3, "<think"))
        return None, "malformed_or_wrapped_json" if shaped else "plain_text"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    return obj, "string_json_object"


def schema7(obj):
    """Frozen key/type/enum rules only; no facts or conditional semantics."""
    problems = []
    missing = sorted(set(FIELDS) - set(obj))
    extra = sorted(set(obj) - set(FIELDS))
    if missing:
        problems.append("missing_keys")
    if extra:
        problems.append("extra_keys")
    if "defect_class" in obj:
        value = obj["defect_class"]
        if not isinstance(value, str):
            problems.append("defect_class_not_string")
        elif value not in CLASSES:
            problems.append("defect_class_out_of_enum")
    if "radial_zone" in obj:
        value = obj["radial_zone"]
        if not isinstance(value, str):
            problems.append("radial_zone_not_string")
        elif value not in ZONES:
            problems.append("radial_zone_out_of_enum")
    if "clock_direction" in obj:
        value = obj["clock_direction"]
        if value is not None and not isinstance(value, str):
            problems.append("clock_direction_wrong_type")
    if "extent_r" in obj and obj["extent_r"] is not None:
        problems.append("extent_r_not_json_null")
    for field in ("morphology", "caption_zh", "uncertainty"):
        if field in obj and not isinstance(obj[field], str):
            problems.append(field + "_not_string")
    return problems, missing, extra


def verify_protocol(root):
    inputs = {}
    for name in (INDEX, PROMPT, PROTOCOL, SCHEMA_SOURCE):
        _, meta = file_snapshot(root / name, root)
        inputs[name] = meta
    protocol = strict_loads((root / PROTOCOL).read_text(encoding="utf-8"))
    prompt = (root / PROMPT).read_text(encoding="utf-8").rstrip("\r\n")
    if sha(protocol["题面"].encode("utf-8")) != EXPECTED_PROMPT_SHA:
        raise ValueError("Frozen protocol question hash changed")
    if protocol["题面_sha256"] != EXPECTED_PROMPT_SHA or prompt != protocol["题面"]:
        raise ValueError("Locked prompt and frozen protocol are not identical")
    if protocol["解析"]["字段"] != list(FIELDS):
        raise ValueError("Frozen field list changed")
    tree = ast.parse((root / SCHEMA_SOURCE).read_text(encoding="utf-8"))
    constants, schema_node = {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("FIELDS", "CLASSES", "ZONES"):
                    constants[target.id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "schema7":
            schema_node = node
    if constants["FIELDS"] != list(FIELDS) or constants["CLASSES"] + ["unknown"] != list(CLASSES):
        raise ValueError("Frozen schema fields/classes changed")
    if constants["ZONES"] != list(ZONES) or schema_node is None:
        raise ValueError("Frozen schema zones changed")
    return inputs, {
        "question_sha256": EXPECTED_PROMPT_SHA,
        "fields_exact": list(FIELDS), "defect_class_enum": list(CLASSES),
        "radial_zone_enum": list(ZONES), "extent_r": "JSON null only",
        "clock_direction": "string or JSON null",
        "text_fields": ["morphology", "caption_zh", "uncertainty"],
        "schema_source": SCHEMA_SOURCE,
        "schema_source_lines": [schema_node.lineno, schema_node.end_lineno],
        "caption_empty": "Separate diagnostic; not a frozen schema7 type failure.",
        "clock_literal_null": "String null passes type check; not equivalent to JSON null.",
        "protocol_implementation_difference": (
            "Protocol prose says out-of-enum radial_zone is counted missing; frozen schema7 "
            "marks it invalid. This audit records invalidity without rewriting values."),
        "limits": [
            "Conditional direction applicability requires semantic/fact checks.",
            "No class-label, geometry, image-content or caption factual comparison is performed.",
            "Duplicate keys, nonfinite JSON, multiple objects and trailing text are rejected.",
        ],
    }


def source_evidence(root):
    base = "协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/"
    scripts = ["save_answer.py", "workers/_finalize_117.py", "workers/_build_120.py",
               "workers/_w114_truth.py", "workers/_w114_step1_hash.py",
               "workers/_w114_verify.py", "workers/rebuild_095.py"]
    results = []
    for name in scripts:
        path = root / (base + name)
        if not path.is_file():
            continue
        content, meta = file_snapshot(path, root)
        source = content.decode("utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        locations = defaultdict(list)
        for line, text in enumerate(source.splitlines(), 1):
            if "PROMPT_SHA =" in text:
                locations["fixed_PROMPT_SHA"].append(line)
            if '"prompt_sha256": PROMPT_SHA' in text:
                locations["writes_fixed_prompt_sha"].append(line)
            if '"check": "self_ok"' in text:
                locations["writes_self_ok_literal"].append(line)
            if "missing = [f for f in fields if f not in obj]" in text:
                locations["checks_missing_keys_only"].append(line)
            if re.match(r"\s*(ANSWERS|ITEMS)\s*=", text):
                locations["static_answer_container"].append(line)
        results.append({**meta, "imports": imports, "evidence_lines": dict(locations)})
    return {
        "scripts": results,
        "limits": [
            "Fixed prompt_sha metadata is not a fingerprint of the actual client invocation.",
            "self_ok is a writer literal; save_answer checks missing keys only.",
            "Static ANSWERS/ITEMS writers hold answer strings, not observable model calls.",
            "The inspected writers do not show PIL/numpy pixel-statistics description generation.",
            "The inspected hash helpers read blind shard indexes and image bytes for SHA checks.",
            "No label/old-teacher/geometry input was found in this limited inspected set.",
            "This limited absence does not prove the client's visual/blind provenance.",
        ],
    }


def audit(root):
    started = now()
    input_files, protocol = verify_protocol(root)
    index_rows, index_problems = saved_jsonl((root / INDEX).read_bytes())
    index = {record["item_id"]: record for _, record in index_rows}
    if index_problems or len(index) != 1200 or len(index_rows) != 1200:
        raise ValueError("Expected 1200 unique complete blind-index rows")
    expected_shard = {identifier: 1 + order // 7 for order, identifier in enumerate(index)}
    directory = root / WORKERS
    files = sorted(path for path in directory.iterdir() if CANONICAL.fullmatch(path.name))
    excluded = [path.name for path in directory.iterdir()
                if path.is_file() and not CANONICAL.fullmatch(path.name)]
    snapshots, rows, outer_problems = [], [], []
    for path in files:
        content, meta = file_snapshot(path, root)
        parsed, problems = saved_jsonl(content)
        meta["complete_records"] = len(parsed)
        snapshots.append(meta)
        outer_problems.extend({"path": meta["path"], **problem} for problem in problems)
        rows.extend((meta["path"], line, record) for line, record in parsed)
    seen, duplicates = {}, []
    for path, line, record in rows:
        identifier = record.get("item_id")
        if not isinstance(identifier, str):
            outer_problems.append({"path": path, "line": line, "code": "item_id_not_string"})
        elif identifier in seen:
            duplicates.append({"item_id": identifier, "path": path, "line": line,
                               "same_raw_answer": record.get("raw_answer") ==
                                                  seen[identifier][2].get("raw_answer")})
        else:
            seen[identifier] = (path, line, record)
    types, parse_codes, buckets, issues = Counter(), Counter(), Counter(), Counter()
    checks, prompt_shas, per_shard = Counter(), Counter(), defaultdict(Counter)
    details, mismatches, shard_mismatches = [], [], []
    empty_captions, clock_literal_null = [], []
    for identifier, (path, line, record) in sorted(seen.items()):
        raw = record.get("raw_answer")
        types[type(raw).__name__] += 1
        obj, parse_code = parse_answer(raw)
        parse_codes[parse_code] += 1
        problems, missing, extra = schema7(obj) if obj is not None else ([], [], [])
        bucket = ("strict_seven_fields" if obj is not None and not problems else
                  "parsed_object_schema_fail" if obj is not None else
                  "plain_text_or_not_unique_json_object")
        buckets[bucket] += 1
        issues.update(problems)
        per_shard[Path(path).name][bucket] += 1
        if parse_code == "plain_text":
            per_shard[Path(path).name]["plain_text"] += 1
        if obj is not None and isinstance(obj.get("caption_zh"), str) and not obj["caption_zh"].strip():
            empty_captions.append(identifier)
        if obj is not None and obj.get("clock_direction") == "null":
            clock_literal_null.append(identifier)
        checks[str(record.get("check", "<missing>"))] += 1
        prompt_shas[str(record.get("prompt_sha256", "<missing>"))] += 1
        expected = index.get(identifier)
        mapped_sample = expected["sample_id"] if expected else None
        if expected:
            if record.get("sample_id") is not None and record["sample_id"] != mapped_sample:
                mismatches.append({"item_id": identifier, "field": "sample_id"})
            if record.get("image_sha256") != expected["image_sha256"]:
                mismatches.append({"item_id": identifier, "field": "image_sha256"})
            actual_shard = int(re.search(r"分片(\d{3})", path).group(1))
            if actual_shard != expected_shard[identifier]:
                shard_mismatches.append(identifier)
        details.append({
            "item_id": identifier, "sample_id_from_blind_index": mapped_sample,
            "source": path, "line": line, "raw_answer_type": type(raw).__name__,
            "parse_code": parse_code, "bucket": bucket, "schema_problems": problems,
            "missing_keys": missing, "extra_keys": extra, "record_check": record.get("check"),
        })
    covered = sorted(set(index) & set(seen))
    files_after = sorted(path.name for path in directory.iterdir()
                         if CANONICAL.fullmatch(path.name))
    changed_after = []
    for meta in snapshots:
        stat = (root / meta["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (meta["bytes_read"], meta["mtime_ns"]):
            changed_after.append(meta["path"])
    new_files = sorted(set(files_after) - {path.name for path in files})
    changed_during = [meta["path"] for meta in snapshots if meta["changed_during_read"]]
    complete = (len(files) == 172 and len(rows) == len(seen) == len(covered) == 1200
                and not outer_problems and not mismatches and not shard_mismatches
                and not changed_during and not changed_after and not new_files)
    output = {
        "audit_version": "1.0", "started_at": started, "finished_at": now(),
        "status": "final_canonical_snapshot" if complete else "in_progress_or_incomplete_snapshot",
        "status_note": (
            "Full 172-shard/1200-item stable disk coverage corroborates the user's completion "
            "statement; this audit does not verify client usage or image provenance."
            if complete else "User completion statement must be separated from incomplete/changing disk coverage."),
        "scope": {
            "canonical_regex": CANONICAL.pattern, "workers": WORKERS,
            "read_only_inputs": True, "network_or_gpu": False, "facts_or_labels_scored": False,
            "tail_policy": "Count newline-terminated successfully parsed outer records only.",
            "excluded_file_count": len(excluded),
            "duplicates_policy": "First canonical occurrence for unique stats; duplicates retained.",
        },
        "frozen_protocol": protocol, "input_fingerprints": input_files,
        "source_files": snapshots,
        "write_activity": {"changed_during_read": changed_during,
                           "changed_after_read": changed_after,
                           "new_canonical_files_after_read": new_files},
        "counts": {
            "canonical_files": len(files), "complete_records": len(rows),
            "unique_item_ids": len(seen), "duplicate_item_extra_rows": len(duplicates),
            "blind_index_rows": 1200, "covered_index_items": len(covered),
            "missing_index_items": len(set(index) - set(seen)),
            "outside_index_items": len(set(seen) - set(index)),
            "unique_sample_ids_mapped": len({index[item]["sample_id"] for item in covered}),
            "unique_image_sha256_mapped": len({index[item]["image_sha256"] for item in covered}),
            "blind_index_unique_image_sha256": len({row["image_sha256"] for row in index.values()}),
            "raw_answer_types": dict(types), "parse_codes": dict(parse_codes),
            "three_way": dict(buckets), "schema_problem_counts": dict(issues),
            "caption_empty_strings": len(empty_captions),
            "clock_direction_literal_null_strings": len(clock_literal_null),
            "metadata_mismatch_count": len(mismatches),
            "shard_assignment_mismatch_count": len(shard_mismatches),
        },
        "missing_item_ids": sorted(set(index) - set(seen)),
        "outside_item_ids": sorted(set(seen) - set(index)),
        "duplicate_records": duplicates, "outer_line_problems": outer_problems,
        "metadata_mismatches": mismatches, "shard_assignment_mismatches": shard_mismatches,
        "plain_text_shards": [name for name in sorted(per_shard) if per_shard[name]["plain_text"]],
        "per_shard_three_way": {key: dict(value) for key, value in sorted(per_shard.items())},
        "writer_metadata": {"check_counts": dict(checks), "prompt_sha256_counts": dict(prompt_shas)},
        "provenance_evidence": source_evidence(root), "record_audit": details,
    }
    assert sum(buckets.values()) == sum(types.values()) == len(seen)
    return output


def self_test():
    good = {"defect_class": "none", "morphology": "", "radial_zone": "none",
            "clock_direction": None, "extent_r": None, "caption_zh": "", "uncertainty": ""}
    assert not schema7(good)[0]
    assert schema7({**good, "extent_r": "null"})[0] == ["extent_r_not_json_null"]
    assert schema7({**good, "defect_class": True})[0] == ["defect_class_not_string"]
    assert schema7({**good, "radial_zone": "全局"})[0] == ["radial_zone_out_of_enum"]
    assert not schema7({**good, "clock_direction": "null"})[0]
    assert "extra_keys" in schema7({**good, "extra": 1})[0]
    assert parse_answer(json.dumps(good))[1] == "string_json_object"
    assert parse_answer('{"defect_class":"none","defect_class":"Loc"}')[1] == "duplicate_key"
    assert parse_answer('{"x":NaN}')[1] == "non_finite"
    assert parse_answer(json.dumps(good) + "{}")[1] == "malformed_or_wrapped_json"
    assert parse_answer(chr(96) * 3 + "json\n{}\n" + chr(96) * 3)[1] == "malformed_or_wrapped_json"
    assert parse_answer("可见少量散点")[1] == "plain_text"
    assert parse_answer([])[1] == "unsupported_raw_type"
    parsed, problems = saved_jsonl(b'{"item_id":"x"}\n{"item_id":"y"}')
    assert len(parsed) == 1 and problems[0]["code"] == "unterminated_tail_not_counted"
    return 14


def render_markdown(result):
    c = result["counts"]
    three = c["three_way"]
    plain = result["plain_text_shards"]
    drift = f"{plain[0]} 至 {plain[-1]}，共 {len(plain)} 片" if plain else "无"
    final = result["status"] == "final_canonical_snapshot"
    rows = [
        "# WorkBuddy 机械快照审计", "",
        f"读取窗口：{result['started_at']} → {result['finished_at']}。",
        "状态：" + ("final：172片/1200项完整且读取稳定，与用户完成声明吻合。" if final
                   else "进行中或磁盘不完整：不能以用户完成声明替代本次磁盘覆盖。"),
        "未调用模型、网络或 GPU，未改写 worker 产物。审计不验证账户用量或事实正确性。", "",
        "| 指标 | 本次值 |", "|---|---:|",
        f"| 规范分片 / 完整记录 | {c['canonical_files']} / {c['complete_records']} |",
        f"| 唯一 item / 映射 sample / 图片 SHA | {c['unique_item_ids']} / {c['unique_sample_ids_mapped']} / {c['unique_image_sha256_mapped']} |",
        f"| 原1200索引覆盖 / 缺口 | {c['covered_index_items']} / {c['missing_index_items']} |",
        f"| 严格七字段类型枚举通过 | {three.get('strict_seven_fields', 0)} |",
        f"| 解析为对象但不合 schema | {three.get('parsed_object_schema_fail', 0)} |",
        f"| 纯文本或未形成唯一 JSON 对象 | {three.get('plain_text_or_not_unique_json_object', 0)} |",
        f"| 其中纯文本 | {c['parse_codes'].get('plain_text', 0)} |",
        f"| 重复 item 额外行 / 元数据不一致 | {c['duplicate_item_extra_rows']} / {c['metadata_mismatch_count']} |",
        "", f"原答类型：{json.dumps(c['raw_answer_types'], ensure_ascii=False)}。",
        f"纯文本分片：{drift}。",
        f"Schema问题：{json.dumps(c['schema_problem_counts'], ensure_ascii=False)}。", "",
        "## 冻结协议与边界", "",
        f"题面 SHA256：{EXPECTED_PROMPT_SHA}。",
        f"协议来源：{PROTOCOL}；实现来源：{SCHEMA_SOURCE}，schema7第107–141行。",
        "七键集合精确一致；类别与径向区按锁定枚举；extent_r为真JSON null；方向为字符串或JSON null；三文本字段必须为字符串。",
        "沿用冻结schema7：空Caption单列，不额外当类型硬失败；方向字符串null类型合法，却不等于JSON null，适用方向语义另验。",
        "协议文字称径向越界计缺失，实际schema7判越界失败。本审计不修改字段值。",
        "严格解析不剥围栏、不截取子串，不接受重复键、NaN/Infinity、多对象或尾随文字。",
        "三分覆盖唯一规范原答；第三类包含纯文本及无法形成唯一对象的情况，细分原因见JSON。",
        "sample/SHA映射仅用无答案索引，未读类别标签、几何gold或图像像素。SHA是索引/记录声明值，未重读PNG。",
        "逐文件指纹、读取变化、缺口、逐条问题及来源行号见同目录审计结果.json。", "",
        "## 写入元数据的证据边界", "",
        "save_answer.py固定写PROMPT_SHA，仅检查缺键；self_ok在后续保存脚本中直接写常量。这些标记不证明真实题面或实际视觉调用。",
        "_finalize_117.py/_build_120.py用静态ANSWERS/ITEMS字串落盘。限定检查中未发现标签、旧teacher或几何读取，未见PIL/numpy统计生成描述。",
        "这不证明模型伪造，也不证明完整会话只凭图作答。实际客户端调用与视觉读取链路未核。",
        "脚本指纹、imports及证据行号见JSON的provenance_evidence。", "",
        "## 不据此宣称的事项", "",
        "机械通过不代表类别、形态、位置、方向或Caption事实正确，也不代表可以直接入训。",
        "纯文本保留原件；若转Caption候选须另记抽取/删改，不补造七字段，不把自报数值当gold。", "",
        "重跑：在仓库根目录执行 python -X utf8 -B 本脚本的仓库相对路径。每次生成新快照；可先加 --self-test 或 --dry-run。", "",
    ]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps({"self_tests_passed": tests}, ensure_ascii=False))
        return
    root = find_root()
    result = audit(root)
    result["self_tests_passed"] = tests
    summary = {"status": result["status"], "counts": result["counts"],
               "write_activity": result["write_activity"]}
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    name = "快照_" + dt.datetime.now(TZ).strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(__file__).resolve().parent / name
    destination.mkdir(exist_ok=False)
    (destination / "审计结果.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "审计汇总.md").write_text(render_markdown(result), encoding="utf-8")
    summary["output"] = relative(destination, root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
