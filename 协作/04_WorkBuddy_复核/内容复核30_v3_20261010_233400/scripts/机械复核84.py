# -*- coding: utf-8 -*-
"""内容复核30轮 · 第1步：84图机械复核（独立现算，只读旧目录，不改动任何旧文件）。

1) 现算 blind84.jsonl 全部84张输入PNG的 SHA256，与清单比对；
2) 复跑 workbuddy_schema_check.py 的 check() 于84份旧 raw（当前时间独立核验）；
3) 输出本次时间的独立核验表（jsonl）与汇总（json）。
不重新看图、不重建当时执行证据、不回填旧时间。
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"D:\pycode\晶圆图研究")
PKG = REPO / "协作" / "01_Codex_指挥" / "无卡CPU整备_20261010"
BLIND = PKG / "WorkBuddy84" / "blind84.jsonl"
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
    rows = []
    items = [json.loads(l) for l in BLIND.read_text(encoding="utf-8").splitlines() if l.strip()]
    for it in items:
        item_id = it["item_id"]
        png = Path(it["image_path"])
        rec = {
            "item_id": item_id,
            "sample_id": it["sample_id"],
            "recorded_image_sha256": it["image_sha256"],
        }
        # 1) 现算PNG SHA
        if png.exists():
            now_sha = sha256_file(png)
            rec["image_exists"] = True
            rec["computed_image_sha256"] = now_sha
            rec["image_sha_match"] = (now_sha == it["image_sha256"])
        else:
            rec["image_exists"] = False
            rec["image_sha_match"] = False
        # 2) 复跑格式检查器（只读旧raw）
        raw = OLD_DIR / f"{item_id}.raw.txt"
        if raw.exists():
            text = raw.read_text(encoding="utf-8")
            res = check(text)
            rec["raw_exists"] = True
            rec["computed_raw_sha256"] = sha256_file(raw)
            rec["schema_ok"] = res["schema_ok"]
            rec["errors"] = res["errors"]
            rec["warnings"] = res["warnings"]
        else:
            rec["raw_exists"] = False
            rec["schema_ok"] = False
            rec["errors"] = ["raw_missing"]
            rec["warnings"] = []
        rows.append(rec)

    n_img_ok = sum(1 for r in rows if r["image_sha_match"])
    n_schema_ok = sum(1 for r in rows if r.get("schema_ok"))
    n_warn = sum(1 for r in rows if r.get("warnings"))
    # raw两两重复审计（现算SHA视角）
    sha_map = {}
    for r in rows:
        sha_map.setdefault(r.get("computed_raw_sha256"), []).append(r["item_id"])
    dup_raw = {k: v for k, v in sha_map.items() if k and len(v) > 1}

    table = OUT / "机械复核84_独立核验表.jsonl"
    with open(table, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "内容复核30轮 第1步：84图机械独立复核（PNG SHA现算 + 检查器复跑）",
        "planned": len(rows),
        "image_exists": sum(1 for r in rows if r["image_exists"]),
        "image_sha_match": n_img_ok,
        "raw_exists": sum(1 for r in rows if r.get("raw_exists")),
        "schema_ok": n_schema_ok,
        "schema_fail": [r["item_id"] for r in rows if r.get("schema_ok") is False],
        "rows_with_warnings": n_warn,
        "duplicate_raw_sha_groups": dup_raw,
        "old_dir_read_only": str(OLD_DIR),
        "note": "仅机械核验：不证明当时真实读图/后端请求，不重建执行时证据；本次时间戳为本表生成时间。",
    }
    (OUT / "机械复核84_汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
