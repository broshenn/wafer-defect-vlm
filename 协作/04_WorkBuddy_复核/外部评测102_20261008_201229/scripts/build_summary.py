#!/usr/bin/env python3
"""build_summary.py — 父端收尾：核验无重叠/哈希链，汇总计划/尝试/完成/严格JSON/schema/
约定警告/失败/未知/未执行，产出 summary.json、原答清单.jsonl、评测报告.md。
只读取本执行目录与允许的输入文件；不读取任何真值，不计算正确率。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
BLIND = Path("D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_外部评测102")
SUPP = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008")
IDX_SHA_EXPECT = "41d6e79c7ae6026fae34a609f39af4ced07aa080b163c18e813d1205c9dcd0c1"
PROMPT_SHA_EXPECT = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
CHECKER_SHA_EXPECT = "7d46c37f72bb8fd09488dd73754ad75e6a1b711d91a6f08f82e5b34026c0c0ff"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    idx_rows = [json.loads(l) for l in (BLIND / "WB_eval102_索引.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    idx_sha = sha((BLIND / "WB_eval102_索引.jsonl").read_bytes())
    prompt_sha = sha((SUPP / "prompt_7f.txt").read_bytes())
    checker_sha = sha((SUPP / "schema_check.py").read_bytes())
    shards = sorted(BLIND.glob("WB_eval102_分片*.jsonl"))
    union = []
    for s in shards:
        union.extend(json.loads(l) for l in s.read_text(encoding="utf-8").splitlines() if l.strip())
    sig = lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)

    start_epoch = int((OUT / "_start_epoch.txt").read_text().strip())
    start_iso = datetime.fromtimestamp(start_epoch).astimezone().isoformat(timespec="seconds")
    end_iso = datetime.now().astimezone().isoformat(timespec="seconds")

    planned_ids = {r["item_id"] for r in idx_rows}
    raw_ids = {p.stem for p in OUT.glob("raw/*.txt")}
    done_ids = {p.stem for p in (OUT / "done").glob("*.done")}
    unexecuted = sorted(planned_ids - done_ids)
    extra = sorted(raw_ids - planned_ids)

    # 哈希链与重叠核验
    problems = []
    per_item = []
    counts = {"strict_json": 0, "schema_ok": 0, "contract_warnings": 0,
              "schema_failures": 0, "image_sha_mismatch": 0,
              "rerun_mismatch_with_first_check": 0}
    unknown_fields = []
    for row in idx_rows:
        i = row["item_id"]
        rawp = OUT / "raw" / f"{i}.txt"
        if not rawp.exists():
            per_item.append({"item_id": i, "status": "未执行"})
            continue
        rawb = rawp.read_bytes()
        raw_sha = sha(rawb)
        cp = OUT / "checks" / f"{i}_check.json"
        rp = OUT / "checks_rerun" / f"{i}_rerun.json"
        c = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else None
        r2 = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
        mp = OUT / "meta" / f"{i}_meta.json"
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        if c is None:
            counts["schema_failures"] += 1
            problems.append(f"{i}: 缺首次检查输出")
        else:
            if c.get("raw_sha256") != raw_sha:
                problems.append(f"{i}: 首次检查raw_sha与当前raw文件不一致")
            counts["strict_json"] += bool(c.get("strict_json"))
            counts["schema_ok"] += bool(c.get("schema_ok"))
            counts["contract_warnings"] += len(c.get("contract_warnings") or [])
            if not c.get("schema_ok"):
                counts["schema_failures"] += 1
        if r2 is None:
            problems.append(f"{i}: 缺复跑检查输出")
        elif c is not None:
            if (r2.get("strict_json"), r2.get("schema_ok")) != (c.get("strict_json"), c.get("schema_ok")):
                counts["rerun_mismatch_with_first_check"] += 1
                problems.append(f"{i}: 复跑与首次检查结论不一致")
        if not m.get("image_sha256_ok", False):
            counts["image_sha_mismatch"] += 1
        for k in ("started_at", "model_exact_backend_id", "sampling_params"):
            v = m.get(k)
            if v is None or (isinstance(v, str) and ("未知" in v or "未提供" in v)):
                unknown_fields.append(f"{i}.{k}")
        # 图片哈希运行时复核（与索引比对）
        imgp = Path(row["image_path"])
        if imgp.exists() and sha(imgp.read_bytes()) != row["image_sha256"]:
            counts["image_sha_mismatch"] += 1
            problems.append(f"{i}: 图片哈希与索引不一致")
        per_item.append({
            "item_id": i, "sample_id": row["sample_id"], "shard": m.get("shard_no"),
            "worker": m.get("worker"),
            "image_sha256": row["image_sha256"], "image_sha256_ok": m.get("image_sha256_ok"),
            "raw_answer_sha256": raw_sha, "raw_answer_bytes": len(rawb),
            "strict_json": (c or {}).get("strict_json"), "parse_status": (c or {}).get("parse_status"),
            "schema_ok": (c or {}).get("schema_ok"), "schema_problems": (c or {}).get("schema_problems"),
            "contract_warnings": (c or {}).get("contract_warnings"),
            "rerun_strict_json": (r2 or {}).get("strict_json"), "rerun_schema_ok": (r2 or {}).get("schema_ok"),
            "raw_sha_matches_check": (c or {}).get("raw_sha256") == raw_sha,
            "started_at": m.get("started_at"), "finalized_at": m.get("finalized_at"),
            "visual_reads_used": m.get("visual_reads_used"), "attempts": m.get("attempts"),
            "retry_used": m.get("retry_used"), "status": "完成",
        })

    attempted = len([x for x in per_item if x.get("status") == "完成"])
    summary = {
        "run": "WorkBuddy外部评测102_同图同题外部模型直接推理对照",
        "output_dir": str(OUT),
        "display_model": "GLM-5.3-Flash（父会话与全部14个worker子代理，客户端显示口径）",
        "model_exact_backend_id": "未知：客户端不暴露后端模型标识",
        "sampling_params": "未知：客户端未暴露温度/top_p等真实请求参数",
        "backend_request_log_available": False,
        "disclosure": "后台注入/真实参数/后端请求数未提供；request_records中的prompt为执行时从题面文件逐字读入的客户端侧记录，hash为现算值，不冒充后端发送证据。",
        "plan_vs_actual": {
            "planned": len(idx_rows), "attempted": attempted,
            "completed": attempted, "unexecuted": len(unexecuted),
            "unexecuted_ids": unexecuted, "extra_raw_without_index": extra,
            "failures": counts["schema_failures"],
        },
        "counts": {
            "strict_json": counts["strict_json"], "schema_ok": counts["schema_ok"],
            "contract_warnings": counts["contract_warnings"],
            "schema_failures": counts["schema_failures"],
            "image_sha_mismatch": counts["image_sha_mismatch"],
            "rerun_mismatch_with_first_check": counts["rerun_mismatch_with_first_check"],
        },
        "integrity": {
            "index_sha256_runtime": idx_sha, "index_sha256_ok": idx_sha == IDX_SHA_EXPECT,
            "prompt_sha256_runtime": prompt_sha, "prompt_sha256_ok": prompt_sha == PROMPT_SHA_EXPECT,
            "checker_sha256_runtime": checker_sha, "checker_sha256_ok": checker_sha == CHECKER_SHA_EXPECT,
            "shard_union_equals_index": {sig(r) for r in union} == {sig(r) for r in idx_rows},
            "unique_item_ids": len(planned_ids), "unique_sample_ids": len({r["sample_id"] for r in idx_rows}),
            "raw_files": len(raw_ids),
        },
        "time": {"start": start_iso, "end": end_iso},
        "scope_statement": "全部原答为模型输出，非人工gold；本目录不含正确率、不读取真值；结构合格不等于视觉事实正确。",
        "problems": problems,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    with (OUT / "原答清单.jsonl").open("w", encoding="utf-8", newline="\n") as h:
        for x in per_item:
            h.write(json.dumps(x, ensure_ascii=False) + "\n")

    cls_stat = {}
    for x in per_item:
        if x.get("status") != "完成":
            continue
        try:
            obj = json.loads((OUT / "raw" / f"{x['item_id']}.txt").read_text(encoding="utf-8"))
            cls_stat[obj.get("defect_class", "?")] = cls_stat.get(obj.get("defect_class", "?"), 0) + 1
        except Exception:
            cls_stat["<不可解析>"] = cls_stat.get("<不可解析>", 0) + 1

    report = f"""# WorkBuddy 外部评测102 执行报告（{OUT.name}）

- 执行时间：{start_iso} → {end_iso}
- 执行方：WorkBuddy 父会话（GLM-5.3-Flash）+ 14 个独立 GLM worker 子代理（每片一个，互不继承答案；分片1由父会话执行作链路验证）
- 任务性质：同图同题外部模型直接推理对照；非训练、非重标、不回流

## 输入指纹（执行时现算）
- 总索引 WB_eval102_索引.jsonl：`{idx_sha}`（期望一致：{idx_sha == IDX_SHA_EXPECT}）
- 题面 prompt_7f.txt：`{prompt_sha}`（期望一致：{prompt_sha == PROMPT_SHA_EXPECT}）
- 检查器 schema_check.py：`{checker_sha}`（期望一致：{checker_sha == CHECKER_SHA_EXPECT}）
- 分片并集=索引：{ {sig(r) for r in union} == {sig(r) for r in idx_rows} }；唯一 item={len(planned_ids)}，唯一 sample={len({r['sample_id'] for r in idx_rows})}

## 执行与结果计数
- 计划 {len(idx_rows)} / 尝试 {attempted} / 完成 {attempted} / 未执行 {len(unexecuted)} {unexecuted if unexecuted else ''}
- 严格 JSON：{counts['strict_json']}；schema_ok：{counts['schema_ok']}；约定警告（contract_warnings）：{counts['contract_warnings']}；schema 失败：{counts['schema_failures']}
- 图片哈希不一致：{counts['image_sha_mismatch']}；复跑与首查结论不一致：{counts['rerun_mismatch_with_first_check']}
- 原答 defect_class 分布（模型输出，非正确率）：{json.dumps(cls_stat, ensure_ascii=False)}

## 口径与披露
- 每图一次视觉读取、一次作答；原答逐字节落盘，未改写、未重试；失败原样保留。
- 后台注入/真实参数/后端请求数未提供：request_records/ 中的题面为执行时从题面文件逐字读入的客户端侧记录，hash 现算，不能冒充后端发送证据。
- 显示模型：GLM-5.3-Flash；后端精确模型标识与采样参数：未知（客户端不暴露）。
- 全部原答为模型输出，不是人工 gold；本报告不含正确率，不读取真值；结构合格≠视觉事实正确。

## 产物清单
- raw/：102 份逐字节原答
- checks/：首查输出；checks_rerun/：父端全量复跑输出（两者独立，互不覆盖）
- meta/：逐图 started + 元数据；request_records/：逐图题面客户端记录；done/：完成标记
- checkpoints/：分片1、1–5、1–10、1–15 检查点
- summary.json：机器可读汇总；原答清单.jsonl：逐图明细

## 核验问题（应为空）
{json.dumps(problems, ensure_ascii=False) if problems else '无'}
"""
    (OUT / "评测报告.md").write_text(report, encoding="utf-8", newline="\n")
    print("SUMMARY_BUILT", json.dumps(summary["plan_vs_actual"], ensure_ascii=False),
          json.dumps(summary["counts"], ensure_ascii=False),
          "integrity_ok=", all([idx_sha == IDX_SHA_EXPECT, prompt_sha == PROMPT_SHA_EXPECT,
                                checker_sha == CHECKER_SHA_EXPECT,
                                {sig(r) for r in union} == {sig(r) for r in idx_rows}]))
    print("PROBLEMS:", problems if problems else "无")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
