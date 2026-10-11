"""固定现有36图并去标签；只生成新的盲索引，不发送请求。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "协作/02_ClaudeCode_实操/部署原型_20261009_0120/样本清单.json"
IMAGES = SOURCE.parent / "图"
OUT = Path(__file__).parent
rows = json.loads(SOURCE.read_text(encoding="utf-8"))["样本"]
assert len(rows) == 36 and len({r["sample_id"] for r in rows}) == 36
blind = []
for i, row in enumerate(rows, 1):
    path = IMAGES / (row["sample_id"] + ".png")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["图sha256"]
    blind.append({"item_id": f"api36_{i:03d}", "sample_id": row["sample_id"],
                  "image_path": str(path.relative_to(ROOT)).replace(chr(92), "/"),
                  "image_sha256": row["图sha256"]})
target = OUT / "blind36.jsonl"
data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in blind).encode("utf-8")
if target.exists() and target.read_bytes() != data:
    raise SystemExit("已冻结内容不同，拒绝覆盖")
target.write_bytes(data)
result = {"n": 36, "unique_sample": 36, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "blind_sha256": hashlib.sha256(data).hexdigest(), "keys": list(blind[0]),
          "scope": "原36图开发对照；去公开类别/lot/来源/学生答案；没有API调用"}
(OUT / "盲包核验.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
