# -*- coding: utf-8 -*-
"""争议审阅卡 · 清单生成与独立核验（修正后的写入顺序：文件先全部完成→清单→核验）。

用法：
  python 生成清单与核验.py --emit-manifest   # 全部文件完成后执行
  python 生成清单与核验.py                   # 独立核验清单 + 原件指纹抽查
清单范围：本目录全部文件；清单外仅 清单自身 与 核验结果.json（按设计后产出）。
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).parent
R30 = D.parent / "内容复核30_v3_20261010_233400"
MANIFEST = D / "新清单_sha256.txt"
RESULT = D / "核验结果.json"
EXCLUDE = {"新清单_sha256.txt", "核验结果.json"}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def emit_manifest():
    lines = [
        "# 新清单_sha256.txt —— 争议审阅卡_20261011_124119 全目录",
        f"# 生成：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}（UTC）",
        "# 规则：12卡、00_索引、会话账目、生成程序先全部完成，本清单最后生成；生成后不重写任何被包含文件。",
        "# 清单外文件（有意排除）：新清单_sha256.txt（自身）；核验结果.json（按设计在清单之后生成）。",
        "# 格式：<sha256>  <相对路径>",
        "",
    ]
    files = sorted(p for p in D.rglob("*") if p.is_file() and p.name not in EXCLUDE)
    for p in files:
        lines.append(f"{sha256_file(p)}  {p.relative_to(D).as_posix()}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest emitted: {len(files)} entries")

def verify():
    out = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    entries, bad = [], []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, rel = line.split("  ", 1)
        p = D / rel
        if not p.exists():
            bad.append({"path": rel, "issue": "missing"})
            continue
        ok = sha256_file(p) == sha
        entries.append({"path": rel, "match": ok})
        if not ok:
            bad.append({"path": rel, "issue": "sha_mismatch"})
    out["manifest_verify"] = {"entries": len(entries), "all_match": not bad, "mismatches": bad}

    # 原件指纹抽查（只读）：12卡涉及的Kimi原答 与 10份GLM review —— 证明整理期间原件未变
    items = ["wbv3_002", "wbv3_006", "wbv3_009", "wbv3_018", "wbv3_024", "wbv3_029",
             "wbv3_033", "wbv3_034", "wbv3_038", "wbv3_061", "wbv3_064", "wbv3_082"]
    r2v = {"wbv3_002": "rv30_001", "wbv3_006": "rv30_004", "wbv3_009": "rv30_005",
           "wbv3_024": "rv30_009", "wbv3_029": "rv30_010", "wbv3_033": "rv30_012",
           "wbv3_034": "rv30_013", "wbv3_038": "rv30_015", "wbv3_061": "rv30_025", "wbv3_064": "rv30_027"}
    old = R30.parent / "方向形态84_v3_20261010_200950"
    snaps = {"kimi_raw": {}, "glm_review": {}}
    for it in items:
        snaps["kimi_raw"][f"{it}.raw.txt"] = sha256_file(old / f"{it}.raw.txt")
    for it, rid in r2v.items():
        snaps["glm_review"][f"{rid}.review.json"] = sha256_file(R30 / f"{rid}.review.json")
    out["originals_fingerprint_spotcheck"] = {
        "note": "只读抽查：整理期间被引用的Kimi原答与GLM裁决原件指纹（与卡内记载一致性由逐卡SHA承担）",
        "fingerprints": snaps,
    }
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"manifest": out["manifest_verify"],
                      "spotcheck": {k: len(v) for k, v in snaps.items()}}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    if "--emit-manifest" in sys.argv:
        emit_manifest()
    else:
        verify()
