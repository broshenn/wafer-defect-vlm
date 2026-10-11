"""只读复算三个模型216条诊断，保持主/副解析分列。"""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "协作/02_ClaudeCode_实操/描述事实验收_20261008_1911"
SRC = BASE / "租卡补诊断_20261008_2355"
PARSER = BASE / "D_存在性诊断/解析回归.py"
spec = importlib.util.spec_from_file_location("existence_parser", PARSER)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
summary, fingerprints = {}, {}
for tag in ("Base", "L-N3072-3407", "D-N3072-3407"):
    path = SRC / "原答" / f"{tag}_existence_raw.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 72 and len({(r["sample_id"], r["cond"]) for r in rows}) == 72
    fingerprints[str(path.relative_to(ROOT)).replace(chr(92), "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = {}
    for cond in ("original", "dered", "allred", "noimage"):
        subset = [r for r in rows if r["cond"] == cond]
        assert len(subset) == 18
        counts = Counter()
        for r in subset:
            val, how = p.parse_strict(r["raw"])
            dv, dh = p.parse_diag(r["raw"])
            assert how == r["how"] and dh == r["diag_how"]
            if cond == "noimage":
                counts["object_null"] += val is None and how == "unknown"
                counts["bare_null"] += p.EMPTY_THINK.sub("", r["raw"].strip()) == "null"
                counts["boolean_claim"] += isinstance(val, bool)
            else:
                assert r["gold"] is (False if cond == "dered" else True)
                counts["strict_correct"] += val is r["gold"] and how == "ok"
                counts["diagnostic_correct"] += dv is r["gold"] and dh in ("ok", "ok_str_bool")
                counts["strict_format_failure"] += how != "ok"
                assert r["strict_ok"] == (val is r["gold"] and how == "ok")
                # 源码diag_ok是解析合法，不是回答正确；必须另与gold比较。
                assert r["diag_ok"] == (dh in ("ok", "ok_str_bool"))
        out[cond] = {"n": 18, **dict(counts)}
    summary[tag] = out
result = {"rows": 216, "models": summary, "source_sha256": fingerprints,
          "scope": "18图单一二值事实辅助诊断；不是整体业务能力或根因归因证明"}
target = Path(__file__).with_name("独立复算.json")
if target.exists():
    raise SystemExit("拒绝覆盖原结果")
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
