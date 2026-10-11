# -*- coding: utf-8 -*-
"""内容复核30轮 · 终验与交付汇总。

1) 30图全量机械终验：五件套齐全、observation过schema检查器、SHA三方自洽
   （review内记录值 vs 现算）、observation先于review落盘（mtime）、review字段与七checks齐全；
2) 校验旧84目录未被本轮改动（全部文件mtime早于本轮开始）；
3) 写 raw_sha256_manifest.txt、final_summary.json、30图复核结果一览.md。
"""
import hashlib
import json
import sys
from datetime import datetime, timezone  # 本轮开始时间从输出目录自身元数据取，不硬编码
from pathlib import Path

REPO = Path(r"D:\pycode\晶圆图研究")
PKG = REPO / "协作" / "01_Codex_指挥" / "无卡CPU整备_20261010"
OLD_DIR = REPO / "协作" / "04_WorkBuddy_复核" / "方向形态84_v3_20261010_200950"
OUT = REPO / "协作" / "04_WorkBuddy_复核" / "内容复核30_v3_20261010_233400"

sys.path.insert(0, str(PKG))
from workbuddy_schema_check import check  # noqa: E402

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    alloc = json.loads((OUT / "分片分配.json").read_text(encoding="utf-8"))["order"]
    items = []
    for s in ["shard1", "shard2", "shard3", "shard4", "shard5"]:
        items += json.loads((OUT / f"{s}_条目.json").read_text(encoding="utf-8"))["items"]
    assert len(items) == 30

    per_item, all_ok = [], True
    for it in items:
        rid, iid = it["review_id"], it["item_id"]
        rec = {"review_id": rid, "item_id": iid, "sample_id": it["sample_id"]}
        files = {k: OUT / f"{rid}.{k}" for k in
                 ["started.json", "observation_raw.json", "observation_meta.json",
                  "review.json", "done.json"]}
        missing = [p.name for p in files.values() if not p.exists()]
        failed = (OUT / f"{rid}.failed.json").exists()
        rec["missing"] = missing
        rec["has_failed_marker"] = failed
        if missing or failed:
            rec["ok"] = False
            all_ok = False
            per_item.append(rec)
            continue
        obs_text = files["observation_raw.json"].read_text(encoding="utf-8")
        chk = check(obs_text)
        obs_sha_now = sha256_file(files["observation_raw.json"])
        old_raw = OLD_DIR / f"{iid}.raw.txt"
        old_sha_now = sha256_file(old_raw)
        review = json.loads(files["review.json"].read_text(encoding="utf-8"))
        meta = json.loads(files["observation_meta.json"].read_text(encoding="utf-8"))
        started = json.loads(files["started.json"].read_text(encoding="utf-8"))
        need_fields = {"review_id", "item_id", "sample_id", "image_sha256",
                       "original_raw_sha256", "observation_raw_sha256",
                       "model_display_name", "checks", "overall",
                       "suggested_revision", "uncertainty"}
        checks7 = {"morphology_zh", "main_location_zh", "location_clock",
                   "orientation_clock", "secondary_observations", "caption_zh", "uncertainty"}
        rec.update({
            "obs_schema_ok": chk["schema_ok"],
            "obs_sha_selfconsistent": obs_sha_now == meta.get("observation_raw_sha256") == review.get("observation_raw_sha256"),
            "old_raw_sha_selfconsistent": old_sha_now == review.get("original_raw_sha256"),
            "image_sha_selfconsistent": started.get("image_sha256_computed") == review.get("image_sha256"),
            "obs_before_review_mtime": files["observation_raw.json"].stat().st_mtime <= files["review.json"].stat().st_mtime,
            "review_fields_ok": set(review) == need_fields,
            "checks7_ok": set(review.get("checks", {})) == checks7,
            "overall": review.get("overall"),
        })
        rec["ok"] = all([rec["obs_schema_ok"], rec["obs_sha_selfconsistent"],
                         rec["old_raw_sha_selfconsistent"], rec["image_sha_selfconsistent"],
                         rec["obs_before_review_mtime"], rec["review_fields_ok"], rec["checks7_ok"]])
        all_ok = all_ok and rec["ok"]
        per_item.append(rec)

    # 旧84目录本轮未被改动（以本轮输出目录创建时刻为界）
    round_start_ts = OUT.stat().st_ctime
    old_touched = [p.name for p in OLD_DIR.iterdir()
                   if p.is_file() and p.stat().st_mtime >= round_start_ts]

    # SHA 清单最后生成，覆盖本目录全部产物（清单自身除外）
    manifest_lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "raw_sha256_manifest.txt":
            rel = p.relative_to(OUT).as_posix()
            manifest_lines.append(f"{sha256_file(p)}  {rel}")
    (OUT / "raw_sha256_manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    n_ok = sum(1 for r in per_item if r.get("ok"))
    overall_counts = {}
    for r in per_item:
        overall_counts[r.get("overall")] = overall_counts.get(r.get("overall"), 0) + 1
    summary = {
        "version": "WB84_content_review30",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "planned": 30, "actually_viewed": sum(1 for r in per_item if "missing" in r and not r["missing"]),
            "observation_written": sum(1 for r in per_item if r.get("obs_schema_ok") is not None),
            "review_written": sum(1 for r in per_item if r.get("overall") is not None),
            "failed": sum(1 for r in per_item if r.get("has_failed_marker")),
            "not_executed": 30 - sum(1 for r in per_item if r.get("overall") is not None),
        },
        "mechanical_validation": {"items_ok": n_ok, "all_ok": all_ok},
        "overall_distribution": overall_counts,
        "old_dir_touched_by_this_round": old_touched,
        "model": {"review_round_display": "GLM-5.3-Flash",
                  "note": "会话声明型号；真实后端与采样参数未提供未核实。原84标来源：用户确认界面选择Kimi K3（见 原标来源说明.md，旧meta未改）。",
                  "credits_balance": "未知（客户端不可自查，旧消耗不重置）"},
        "boundaries": [
            "本结果为AI复核，不是人审/专家gold；不得将定向抽查30图的结论推广到全84图。",
            "观察先于旧答打开的客户端证据=文件落盘顺序与各meta时间戳；不冒充后端请求数。",
            "原84答案未修改未覆盖；suggested_revision仅为建议，不自动进入训练。",
        ],
        "mechanical_artifacts": {
            "84机械复核": ["机械复核84_独立核验表.jsonl", "机械复核84_汇总.json"],
            "056更正": "wbv3_056_started更正说明.md",
            "来源说明": "原标来源说明.md",
            "sha清单": "raw_sha256_manifest.txt",
        },
        "per_item": per_item,
    }
    (OUT / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 30图内容复核结果一览（rv30_001..030，按任务书指定顺序）", "",
             f"生成：{summary['generated_at_local']}；分布：usable {overall_counts.get('usable',0)} / requires_review {overall_counts.get('requires_review',0)} / reject {overall_counts.get('reject',0)}；机械终验 {'全过' if all_ok else '存在问题'}", "",
             "| review_id | item_id | sample_id | overall | 一览 |", "|---|---|---|---|---|"]
    for r in per_item:
        rev = json.loads((OUT / f"{r['review_id']}.review.json").read_text(encoding="utf-8")) if r.get("overall") else None
        brief = ""
        if rev:
            contra = [k for k, v in rev["checks"].items() if v["verdict"] == "contradicted"]
            unv = [k for k, v in rev["checks"].items() if v["verdict"] == "unverifiable"]
            parts = []
            if contra: parts.append("contradicted:" + ",".join(contra))
            if unv: parts.append("unverifiable:" + ",".join(unv))
            if rev.get("uncertainty"): parts.append("分歧已记录")
            brief = "；".join(parts) if parts else "七项一致"
        lines.append(f"| {r['review_id']} | {r['item_id']} | {r['sample_id']} | {r.get('overall','—')} | {brief} |")
    (OUT / "30图复核结果一览.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ["counts", "mechanical_validation", "overall_distribution", "old_dir_touched_by_this_round"]}, ensure_ascii=False, indent=1))
    if not all_ok:
        for r in per_item:
            if not r.get("ok"):
                print("ISSUE:", json.dumps(r, ensure_ascii=False))

if __name__ == "__main__":
    main()
