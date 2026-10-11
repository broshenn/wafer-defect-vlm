# -*- coding: utf-8 -*-
"""追加记录 · 程序D：新清单生成与独立核验。

用法：
  python 程序D_独立核验新目录.py --emit-manifest   # 先行：汇总/一览全部完成之后执行，生成 新清单_sha256.txt
  python 程序D_独立核验新目录.py                   # 之后：独立核验（清单逐条比对 + 旧清单陈旧指纹确认 + 原件现状指纹）

清单范围：本目录（追加记录_20261011_0133）全部文件，除 清单自身 与 程序D_核验结果.json（核验结果按设计在清单之后生成，列为清单外文件）。
清单生成后不重写任何被包含文件。
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).parent
ROUND = D.parent  # 内容复核30_v3_20261010_233400
MANIFEST = D / "新清单_sha256.txt"
RESULT = D / "程序D_核验结果.json"
EXCLUDE = {"新清单_sha256.txt", "程序D_核验结果.json"}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def emit_manifest():
    lines = [
        "# 新清单_sha256.txt —— 追加记录_20261011_0133 全目录（修正制作顺序后的清单）",
        f"# 生成：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}（UTC）",
        "# 规则：汇总与一览（终版副本/）及全部说明、程序、程序结果先完成，本清单最后生成；生成后不重写任何被包含文件。",
        "# 清单外文件（有意排除）：新清单_sha256.txt（自身）；程序D_核验结果.json（按设计在清单之后生成）。",
        "# 格式：<sha256>  <相对路径>（相对本目录，POSIX分隔符）",
        "",
    ]
    files = sorted(p for p in D.rglob("*") if p.is_file() and p.name not in EXCLUDE)
    for p in files:
        rel = p.relative_to(D).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest emitted: {len(files)} entries")

def verify():
    out = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "manifest": MANIFEST.name}

    # 1) 新清单逐条核验
    entries, bad = [], []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, rel = line.split("  ", 1)
        p = D / rel
        if not p.exists():
            bad.append({"path": rel, "issue": "missing"})
            continue
        now = sha256_file(p)
        ok = (now == sha)
        entries.append({"path": rel, "manifest_sha": sha, "recomputed": now, "match": ok})
        if not ok:
            bad.append({"path": rel, "issue": "sha_mismatch"})
    # 清单外文件现状
    outside = {name: (sha256_file(D / name) if (D / name).exists() else None) for name in EXCLUDE}
    out["new_manifest_verify"] = {"entries": len(entries), "all_match": not bad,
                                  "mismatches": bad, "outside_files": outside}

    # 2) 旧清单陈旧指纹确认（旧清单在上一级目录，路径相对上一级）
    old_rows, old_bad = [], []
    old_man = ROUND / "raw_sha256_manifest.txt"
    for line in old_man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, rel = line.split("  ", 1)
        p = ROUND / rel
        if not p.exists():
            old_bad.append({"path": rel, "issue": "missing_now"})
            continue
        now = sha256_file(p)
        ok = (now == sha)
        old_rows.append({"path": rel, "match": ok})
        if not ok:
            old_bad.append({"path": rel, "issue": "stale_fingerprint",
                            "manifest_sha": sha, "current_sha": now})
    out["old_manifest_recheck"] = {
        "entries": len(old_rows),
        "mismatch_count": len(old_bad),
        "mismatches": old_bad,
        "explanation": "脚本在同一次运行中先写清单、后重写 final_summary.json 与 30图复核结果一览.md（generated_at 变化），且脚本被重跑过；故旧清单中恰好这两条为上一次写入的陈旧指纹。旧清单/旧汇总原件保留不改。",
    }

    # 3) 原件现状指纹（只读补记，供下游分账核对原件自本轮以来未被改动）
    snaps = {}
    for kind in ["started", "observation_raw", "observation_meta", "review", "done"]:
        snaps[kind] = {}
        for p in sorted(ROUND.glob(f"rv30_*.{kind}.json")):
            snaps[kind][p.name] = sha256_file(p)
    out["originals_current_fingerprints"] = {
        "note": "只读现状指纹（30图×5件），供Claude候选分账前核对原件未被改动；非重新审核。",
        "files": snaps,
        "old_raw_84_note": "旧84目录（方向形态84_v3_20261010_200950）未列入指纹重扫：程序A轮已核旧目录零改动，本轮亦未写入。",
    }
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "new_manifest": {"entries": out["new_manifest_verify"]["entries"], "all_match": out["new_manifest_verify"]["all_match"]},
        "old_manifest": {"entries": out["old_manifest_recheck"]["entries"], "mismatch_count": out["old_manifest_recheck"]["mismatch_count"],
                         "mismatch_paths": [m["path"] for m in out["old_manifest_recheck"]["mismatches"]]},
        "originals": {k: len(v) for k, v in snaps.items()},
    }, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    if "--emit-manifest" in sys.argv:
        emit_manifest()
    else:
        verify()
