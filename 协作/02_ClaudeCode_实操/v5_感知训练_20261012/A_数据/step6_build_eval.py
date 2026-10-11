#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 6 步：评测与 RL 文件（合同 5.4/5.5）。
输入：SELECT200/TEST/TEST200/ROT200/SHUF100.txt、geom_all.jsonl、TRAIN_POOL.txt、matrices/、img448/、prompt_v5.txt
输出（全部 __IMG__/ 占位路径）：
  select200_main、test_main、test200_main、rot200_main、rot100_main、shuf100_main、shuf50_main（各含 _x2 共 14 个）
  select200_L、test_L、rl_candidates800（320 行）、rsft_rows_template（1280 行）
  ROT 图 img448/<id>_r090.png（TEST 图顺时针 90°）；SHUF 图 img448/<id>_shuf.png（同一像素置换）
校验：行数；每个评测文件与 TRAIN_POOL 隔离 0；rl_candidates in TRAIN_POOL；reference_json 过 schema。
"""
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import syn_lib  # noqa: E402
from isolation_check import check, check_subset  # noqa: E402

L_SRC = REPO / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/L_N3072.jsonl"
WAFER_PNG_V2 = REPO / "协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/wafer_png_v2.py"


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def strat_sample(ids, k, seed, meta, min_per_class=5):
    groups = {}
    for s in ids:
        groups.setdefault(meta[s]["failure_type"], []).append(s)
    counts = {c: len(v) for c, v in groups.items()}
    total = sum(counts.values())
    k = min(k, total)
    floor_min = {c: min(min_per_class, counts[c]) for c in counts}
    alloc = {c: max(floor_min[c], min(counts[c], round(k * counts[c] / total))) for c in counts}
    order_hi = sorted(counts, key=lambda c: (-(alloc[c] - floor_min[c]), c))
    while sum(alloc.values()) > k:
        for c in order_hi:
            if alloc[c] > floor_min[c]:
                alloc[c] -= 1
                break
    order_lo = sorted(counts, key=lambda c: (-(counts[c] - alloc[c]), c))
    while sum(alloc.values()) < k:
        for c in order_lo:
            if alloc[c] < counts[c]:
                alloc[c] += 1
                break
    rng = random.Random(seed)
    picked = []
    for c in sorted(counts):
        picked.extend(rng.sample(sorted(groups[c]), alloc[c]))
    return sorted(picked)


def main():
    core = syn_lib.load_core()
    wpng_spec = importlib.util.spec_from_file_location("_wafer_png_v2", WAFER_PNG_V2)
    wpng = importlib.util.module_from_spec(wpng_spec)
    wpng_spec.loader.exec_module(wpng)
    prompt = (HERE / "prompt_v5.txt").read_text(encoding="utf-8")
    geo = {r["sample_id"]: r for r in load_jsonl(HERE / "geom_all.jsonl")}
    meta = {sid: {"lot_name": r["lot_name"], "png_sha256": r["png_sha256"]} for sid, r in geo.items()}
    pool_ids = (HERE / "TRAIN_POOL.txt").read_text(encoding="utf-8").split()
    pool_set = set(pool_ids)
    select200 = (HERE / "SELECT200.txt").read_text(encoding="utf-8").split()
    test_ids = (HERE / "TEST.txt").read_text(encoding="utf-8").split()
    test200 = (HERE / "TEST200.txt").read_text(encoding="utf-8").split()
    rot200 = (HERE / "ROT200.txt").read_text(encoding="utf-8").split()
    shuf100 = (HERE / "SHUF100.txt").read_text(encoding="utf-8").split()
    rot100 = rot200[:100]
    shuf50 = shuf100[:50]
    L_prompt = json.loads(L_SRC.read_text(encoding="utf-8").splitlines()[0])["messages"][0]["content"]
    assert L_prompt.startswith("<image>")

    cache = {}

    def geom_full(cache_key, cls, matrix=None):
        """返回完整 reference 字典（含未舍入值）。cache_key 区分同 id 的不同派生图。"""
        if cache_key in cache:
            return cache[cache_key]
        m = matrix if matrix is not None else np.load(HERE / "matrices" / f"{cache_key}.npy")
        g = core.measure_v3(m)
        assert g["status"] in ("ok", "no_red") and not g["outline"]["anomalous"], cache_key
        n_red = int(g["n_red"])
        n_valid = int(g["n_green"]) + n_red
        ratio_exact = (n_red / n_valid) if n_valid else 0.0
        zf_exact = ([g["radial"]["zone_mass"][z] / n_red for z in ("center", "middle", "edge")]
                    if n_red else [0.0, 0.0, 0.0])
        r1 = float(g["direction"].get("R1_concentration", 0.0))
        r2 = float(g["direction"].get("R2_axial", 0.0))
        # 目标值（合同 3.3）：red_ratio 用 g["coverage"] 两位；zone_frac 用 g 的 5 位源值一位；R1/R2 一位
        rounded = {"red_ratio": float(f"{g['coverage']:.2f}"),
                   "zone_frac": [float(f"{g['radial']['zone_frac'][z]:.1f}") for z in ("center", "middle", "edge")],
                   "R1": float(f"{r1:.1f}"), "R2": float(f"{r2:.1f}")}
        exact = {"red_ratio": ratio_exact, "zone_frac": zf_exact, "R1": r1, "R2": r2}
        ref = {"defect_class": cls,
               "coverage_band": core.COV_MAP[g["coverage_level"]],
               "red_mass_zones": g["red_mass_zones"],
               "angular_type": core.ANG_MAP[g["direction_type"]],
               "clock_sectors": list(g["clock_sectors"]),
               "uncertainty": ("全红点角向分布证据不足" if g["direction_type"] == "unknown" else ""),
               "measure": rounded, "measure_exact": exact}
        cache[cache_key] = ref
        return ref

    def ref_row(cache_key, cls, split_src, lot_sid, img_name=None, extra=None, matrix=None):
        ref = dict(geom_full(cache_key, cls, matrix))
        ref["split_src"] = split_src
        ref["lot_name"] = geo[lot_sid]["lot_name"]
        row = {"messages": [{"role": "user", "content": "<image>\n" + prompt}],
               "images": [f"__IMG__/{img_name or cache_key}.png"], "sample_id": lot_sid if extra and extra.get("src_id") else cache_key,
               "reference_json": json.dumps(ref, ensure_ascii=False, separators=(",", ":"))}
        if extra:
            row.update(extra)
        return row

    def write_rows(name, rows):
        p = HERE / name
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        return hashlib.sha256(p.read_bytes()).hexdigest()

    shas = {}

    # ── 1. 主题面评测文件 ──
    select_rows = [ref_row(sid, geo[sid]["failure_type"], "val", sid) for sid in select200]
    test_rows = [ref_row(sid, geo[sid]["failure_type"], "test", sid) for sid in test_ids]
    test200_rows = [ref_row(sid, geo[sid]["failure_type"], "test", sid) for sid in test200]
    shas["select200_main.jsonl"] = write_rows("select200_main.jsonl", select_rows)
    shas["test_main.jsonl"] = write_rows("test_main.jsonl", test_rows)
    shas["test200_main.jsonl"] = write_rows("test200_main.jsonl", test200_rows)

    # ── 2. ROT：TEST 图顺时针 90°，重新渲染与重算（几何键 = <sid>_r090）──
    rot_rows = []
    for sid in rot200:
        m = np.load(HERE / "matrices" / f"{sid}.npy")
        m_rot = np.rot90(m, k=-1)
        wpng.matrix_to_image(m_rot, 448).save(HERE / "img448" / f"{sid}_r090.png",
                                              format="PNG", optimize=True)
        rot_rows.append(ref_row(f"{sid}_r090", geo[sid]["failure_type"], "rot", sid,
                                img_name=f"{sid}_r090", extra={"src_id": sid, "rot_deg": 90}, matrix=m_rot))
    shas["rot200_main.jsonl"] = write_rows("rot200_main.jsonl", rot_rows)
    rot_row_by_sid = {r["sample_id"]: r for r in rot_rows}
    shas["rot100_main.jsonl"] = write_rows("rot100_main.jsonl", [rot_row_by_sid[s] for s in rot100])

    # ── 3. SHUF：同一像素置换；reference 仍用原图几何 ──
    perm = np.random.default_rng(3407).permutation(448 * 448)
    shuf_rows = []
    for sid in shuf100:
        arr = np.array(Image.open(HERE / "img448" / f"{sid}.png").convert("RGB"))
        arr2 = arr.reshape(-1, 3)[perm].reshape(448, 448, 3)
        Image.fromarray(arr2).save(HERE / "img448" / f"{sid}_shuf.png", format="PNG", optimize=True)
        shuf_rows.append(ref_row(sid, geo[sid]["failure_type"], "shuf", sid,
                                 img_name=f"{sid}_shuf", extra={"src_id": sid, "shuf_perm_seed": 3407}))
    shas["shuf100_main.jsonl"] = write_rows("shuf100_main.jsonl", shuf_rows)
    shuf_row_by_sid = {r["sample_id"]: r for r in shuf_rows}
    shas["shuf50_main.jsonl"] = write_rows("shuf50_main.jsonl", [shuf_row_by_sid[s] for s in shuf50])

    # ── 4. _x2 版本（images 文件名加 _x2，其余相同）──
    for base in ("select200_main", "test_main", "test200_main", "rot200_main", "rot100_main",
                 "shuf100_main", "shuf50_main"):
        rows = load_jsonl(HERE / f"{base}.jsonl")
        for r in rows:
            r["images"][0] = r["images"][0].replace(".png", "_x2.png")
        shas[f"{base}_x2.jsonl"] = write_rows(f"{base}_x2.jsonl", rows)

    # ── 5. L 题面 ──
    select_L = [{"messages": [{"role": "user", "content": L_prompt}],
                 "images": [f"__IMG__/{sid}.png"], "sample_id": sid,
                 "label": geo[sid]["failure_type"]} for sid in select200]
    test_L = [{"messages": [{"role": "user", "content": L_prompt}],
               "images": [f"__IMG__/{sid}.png"], "sample_id": sid,
               "label": geo[sid]["failure_type"]} for sid in test_ids]
    shas["select200_L.jsonl"] = write_rows("select200_L.jsonl", select_L)
    shas["test_L.jsonl"] = write_rows("test_L.jsonl", test_L)

    # ── 6. RL 候选 320 与 R-SFT 模板 ──
    rl_ids = strat_sample(sorted(pool_set), 320, 3407, geo)
    rl_rows = [ref_row(sid, geo[sid]["failure_type"], "train", sid) for sid in rl_ids]
    shas["rl_candidates800.jsonl"] = write_rows("rl_candidates800.jsonl", rl_rows)

    def main_answer(sid):
        r = geo[sid]
        z = r["zone_frac"]
        line1 = (f"<measure>red_ratio={r['coverage']:.2f}; zone_frac={z[0]:.1f},{z[1]:.1f},{z[2]:.1f};"
                 f" R1={r['R1']:.1f}; R2={r['R2']:.1f}</measure>")
        obj = {"defect_class": r["failure_type"], "coverage_band": r["coverage_band"],
               "red_mass_zones": r["red_mass_zones"], "angular_type": r["angular_type"],
               "clock_sectors": r["clock_sectors"],
               "uncertainty": ("全红点角向分布证据不足" if r["angular_type"] == "unknown" else "")}
        return line1 + "\n" + json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    rsft_rows = []
    for sid in rl_ids:
        ans = main_answer(sid)
        for _ in range(4):
            rsft_rows.append({"messages": [{"role": "user", "content": "<image>\n" + prompt},
                                           {"role": "assistant", "content": ans}],
                              "images": [f"__IMG__/{sid}.png"], "sample_id": sid,
                              "task": "main", "line_src": "real"})
    shas["rsft_rows_template.jsonl"] = write_rows("rsft_rows_template.jsonl", rsft_rows)

    # ── 7. 校验 ──
    checks = []
    expect = {"select200_main.jsonl": 200, "test_main.jsonl": len(test_ids),
              "test200_main.jsonl": 200, "rot200_main.jsonl": 200, "rot100_main.jsonl": 100,
              "shuf100_main.jsonl": 100, "shuf50_main.jsonl": 50,
              "select200_L.jsonl": 200, "test_L.jsonl": len(test_ids),
              "rl_candidates800.jsonl": 320, "rsft_rows_template.jsonl": 1280}
    for n, k in expect.items():
        cnt = sum(1 for _ in open(HERE / n, encoding="utf-8"))
        checks.append((f"{n} 行数={k}", cnt == k, cnt))
    train_pool_list = sorted(pool_set)
    r_all = check(train_pool_list, [r["sample_id"] for r in select_rows] + test_ids, meta,
                  "TRAIN_POOL", "SELECT∪TEST")
    checks.append(("TRAIN_POOL vs SELECT∪TEST 隔离", r_all["id_overlap"] == 0, r_all))
    for n in ("rot200_main.jsonl", "rot100_main.jsonl", "shuf100_main.jsonl", "shuf50_main.jsonl"):
        ids = [r["sample_id"] for r in load_jsonl(HERE / n)]
        rr = check(train_pool_list, ids, meta, "TRAIN_POOL", n)
        checks.append((f"{n} 与 TRAIN_POOL 隔离", rr["id_overlap"] == 0 and rr["lot_overlap"] == 0, rr["n_b"]))
    sub = check_subset(rl_ids, train_pool_list, meta, "rl_candidates", "TRAIN_POOL")
    checks.append(("rl_candidates in TRAIN_POOL", sub["outside"] == 0, sub["n_sub"]))
    n_schema_bad = 0
    for n in ("select200_main.jsonl", "test_main.jsonl", "test200_main.jsonl", "rot200_main.jsonl",
              "rl_candidates800.jsonl"):
        for r in load_jsonl(HERE / n):
            try:
                obj = core.parse(r["reference_json"])
                six = {k: obj[k] for k in core.FIELDS}   # 合同 5.5：六字段是子集，另含 measure 等扩展字段
                if core.schema_errors(six):
                    n_schema_bad += 1
            except Exception:
                n_schema_bad += 1
    checks.append(("reference_json 六字段全部过 schema（含扩展字段）", n_schema_bad == 0, n_schema_bad))
    for name, ok, info in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name} | {info}")

    print("RL 类别分布:", dict(Counter(geo[s]["failure_type"] for s in rl_ids)))
    print("样例 select200_main[0]:", json.dumps(select_rows[0], ensure_ascii=False)[:420])
    print("样例 rot200_main[0]:", json.dumps(rot_rows[0], ensure_ascii=False)[:380])
    print("样例 select200_L[0]:", json.dumps(select_L[0], ensure_ascii=False)[:260])
    manifest = HERE / "数据清单.json"
    items = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    items = [it for it in items if it["file"] not in shas]
    for n, sha in shas.items():
        cnt = expect.get(n) or sum(1 for _ in open(HERE / n, encoding="utf-8"))
        items.append({"file": n, "lines": cnt, "sha256": sha})
    manifest.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
