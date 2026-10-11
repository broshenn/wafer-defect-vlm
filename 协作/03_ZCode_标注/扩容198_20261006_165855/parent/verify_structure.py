# -*- coding: utf-8 -*-
"""父会话结构核验：分片hash/四字段/ID唯一/并集一致/PNG指纹。只读输入，报告写本人目录。"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(r"D:\pycode\晶圆图研究")
IN = ROOT / "协作" / "02_ClaudeCode_实操" / "ZCode标注准备_20261006"
OUT = ROOT / "协作" / "03_ZCode_标注" / "扩容198_20261006_165855" / "parent"

report = []
def say(msg):
    print(msg)
    report.append(str(msg))

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key: {k}")
        d[k] = v
    return d

def load_json_strict(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=no_dup_pairs)

ok = True

# 1. 索引
idx_path = IN / "zcode_input_index.json"
idx = load_json_strict(idx_path)
say(f"index_sha256={sha256_file(idx_path)}")
say(f"index_n_items={idx['n_items']} index_items_len={len(idx['items'])} index_batches={len(idx['batches'])}")

# 2. 27个分片
FIELDS = {"item_id", "sample_id", "image_path", "image_sha256"}
all_items = {}
for b in idx["batches"]:
    bp = IN / b["file"]
    h = sha256_file(bp)
    if h != b["sha256"]:
        say(f"FAIL hash {b['file']} got={h} want={b['sha256']}")
        ok = False
        continue
    lines = [l for l in bp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) != b["n"]:
        say(f"FAIL count {b['file']} lines={len(lines)} want={b['n']}")
        ok = False
    bad_field = 0
    for i, l in enumerate(lines, 1):
        try:
            o = json.loads(l, object_pairs_hook=no_dup_pairs)
        except Exception as e:
            say(f"FAIL json {b['file']}:{i} {e}")
            ok = False
            bad_field += 1
            continue
        if set(o.keys()) != FIELDS:
            say(f"FAIL fields {b['file']}:{i} keys={sorted(o.keys())}")
            ok = False
            bad_field += 1
            continue
        it = (o["item_id"], o["sample_id"], o["image_path"], o["image_sha256"])
        if o["item_id"] in all_items:
            say(f"FAIL dup item_id {o['item_id']} in {b['file']}:{i}")
            ok = False
        else:
            all_items[o["item_id"]] = it
    say(f"OK {b['file']} n={b['n']} hash_match=1 bad_lines={bad_field}")

# 3. blind_new
bn_path = IN / "zcode_blind_new.jsonl"
say(f"blind_new_sha256={sha256_file(bn_path)}")
bn_lines = [l for l in bn_path.read_text(encoding="utf-8").splitlines() if l.strip()]
bn_items = {}
for i, l in enumerate(bn_lines, 1):
    o = json.loads(l, object_pairs_hook=no_dup_pairs)
    if set(o.keys()) != FIELDS:
        say(f"FAIL blind_new fields line {i}: {sorted(o.keys())}")
        ok = False
        continue
    bn_items[o["item_id"]] = (o["item_id"], o["sample_id"], o["image_path"], o["image_sha256"])
say(f"blind_new_lines={len(bn_lines)}")

# 4. 并集一致
idx_items = {it["item_id"]: (it["item_id"], it["sample_id"], it["image_path"], it["image_sha256"]) for it in idx["items"]}
say(f"batch_union={len(all_items)} blind_new_n={len(bn_items)} index_items_n={len(idx_items)}")
same_keys = set(all_items) == set(bn_items) == set(idx_items)
say(f"keys_identical={same_keys}")
if same_keys:
    diff = [k for k in all_items if all_items[k] != bn_items[k] or all_items[k] != idx_items[k]]
    say(f"field_value_mismatches={len(diff)}")
    if diff:
        ok = False
else:
    ok = False
    only_b = sorted(set(all_items) - set(bn_items) - set(idx_items))[:10]
    only_bn = sorted(set(bn_items) - set(all_items))[:10]
    only_ix = sorted(set(idx_items) - set(all_items))[:10]
    say(f"only_in_batches={only_b} only_in_blind_new={only_bn} only_in_index={only_ix}")

# 5. item_id 序列
expect = [f"zc_{i:03d}" for i in range(1, 199)]
say(f"item_id_seq_zc001_198={sorted(all_items.keys()) == expect}")

# 6. PNG 存在性与指纹
bad_png = []
for k in sorted(all_items):
    _, _, ip, ih = all_items[k]
    p = Path(ip)
    if not p.exists():
        bad_png.append((k, "missing"))
    elif sha256_file(p) != ih:
        bad_png.append((k, "hash_mismatch"))
say(f"png_checked={len(all_items)} png_bad={len(bad_png)}")
for k, why in bad_png:
    say(f"PNG_BAD {k} {why}")
    ok = False

verdict = "PASS" if ok else "FAIL"
say(f"VERDICT={verdict}")
(OUT / "verify_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
sys.exit(0 if ok else 1)
