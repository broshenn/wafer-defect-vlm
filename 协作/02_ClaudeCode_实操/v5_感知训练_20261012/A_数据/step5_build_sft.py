#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 5 步：六个训练数据文件（合同 5.3–5.4）。
输入：geom_all.jsonl、geom_rot.jsonl、geom_syn.jsonl、TRAIN_POOL.txt、prompt_v5.txt、atomic_templates.json、matrices/、data/images
输出：sft_S0/S1/S2/S3/S4/S1p.jsonl（各 12000 行）+ sft_stats.json（+ 追加 数据清单.json）
规则：S2=S1 与 S1p=S1 逐字节相同；S3=S1 仅图片名加 `_x2`；
      S1 = 前 4000 原子问答（真实/旋转 2500 + 合成 1500；模板 T1..T6=600,600,600,600,800,800，
      T6 只从 single/opposed 出）+ 后 8000 主任务；S0/S4 = 12000 主任务（S4 只用原图）。
      打乱：主池抽选用 random.Random(3407)，抽完一轮再抽下一轮；原子段建好后再整体打乱一次。
校验：行数=12000；S1/S2/S1p SHA 相同；S3 与 S1 仅 images 字段差 `_x2`；
      随机 200 条主任务用 core_v3.reference(measure_v3(矩阵)) 重算完全一致。
"""
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import syn_lib  # noqa: E402  # 仅为 sys.path 与 load_core 复用（不改其状态）

SEG_ATOMIC = 4000
SEG_MAIN_S1 = 8000
N_MAIN = 12000
TEMPLATE_COUNTS = {"T1": 600, "T2": 600, "T3": 600, "T4": 600, "T5": 800, "T6": 800}


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def measure_line(r):
    z = r["zone_frac"]
    return (f"<measure>red_ratio={r['coverage']:.2f}; zone_frac={z[0]:.1f},{z[1]:.1f},{z[2]:.1f};"
            f" R1={r['R1']:.1f}; R2={r['R2']:.1f}</measure>")


def json_fields(r, cls):
    obj = {"defect_class": cls, "coverage_band": r["coverage_band"],
           "red_mass_zones": r["red_mass_zones"], "angular_type": r["angular_type"],
           "clock_sectors": r["clock_sectors"],
           "uncertainty": ("全红点角向分布证据不足" if r["angular_type"] == "unknown" else "")}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main_answer(r, cls):
    return measure_line(r) + "\n" + json_fields(r, cls)


def atomic_answer(r, tid):
    if tid == "T1":
        return r["coverage_band"]
    if tid == "T2":
        return f"{r['coverage']:.2f}"
    if tid == "T3":
        z = r["red_mass_zones"]
        return "unknown" if z == "unknown" else json.dumps(z, ensure_ascii=False, separators=(",", ":"))
    if tid == "T4":
        z = r["zone_frac"]
        return f"{z[0]:.1f},{z[1]:.1f},{z[2]:.1f}"
    if tid == "T5":
        return r["angular_type"]
    if tid == "T6":
        return json.dumps(r["clock_sectors"], separators=(",", ":"))
    raise ValueError(tid)


def make_row(img_name, user_text, answer, task, line_src):
    return {"messages": [{"role": "user", "content": "<image>\n" + user_text},
                         {"role": "assistant", "content": answer}],
            "images": [f"__IMG__/{img_name}.png"], "sample_id": img_name,
            "task": task, "line_src": line_src}


def draw_rounds(names, k, rng):
    """按轮次无放回抽取：每轮把池打乱一次取剩余所需。"""
    pool = list(names)
    out = []
    while len(out) < k:
        rng.shuffle(pool)
        out.extend(pool[: k - len(out)])
    return out


def main():
    prompt = (HERE / "prompt_v5.txt").read_text(encoding="utf-8")
    templates = json.loads((HERE / "atomic_templates.json").read_text(encoding="utf-8"))
    pool_ids = (HERE / "TRAIN_POOL.txt").read_text(encoding="utf-8").split()
    pool_set = set(pool_ids)
    geo = {r["sample_id"]: r for r in load_jsonl(HERE / "geom_all.jsonl")}
    rot = {r["sample_id"]: r for r in load_jsonl(HERE / "geom_rot.jsonl")}
    syn = {r["sample_id"]: r for r in load_jsonl(HERE / "geom_syn.jsonl")}
    import importlib.util
    core = syn_lib.load_core()

    def geom_of(img):
        if img.startswith("syn_"):
            return syn[img], None
        if "_r" in img and img.split("_r")[-1] in ("090", "180", "270"):
            r = rot[img]
            return r, geo[r["src_id"]]["failure_type"]
        return geo[img], geo[img]["failure_type"]

    # ── 主池 ──
    real_names = sorted(pool_ids)
    rot_names = sorted(f"{sid}_r{deg:03d}" for sid in pool_ids for deg in (90, 180, 270))
    main_pool = sorted(real_names + rot_names)
    perm = list(main_pool)
    random.Random(3407).shuffle(perm)
    s0_main = perm[:N_MAIN]          # 12000 < 18144，一轮
    s1_main = perm[:SEG_MAIN_S1]     # 与 S0 同一排列的前 8000
    # S4：只有原图，按轮次重复抽取（原子 4000 + 主任务 8000 = 12000）
    s4_main = draw_rounds(real_names, SEG_MAIN_S1, random.Random(3407))

    def build_main_rows(order):
        rows = []
        for img in order:
            r, cls = geom_of(img)
            src = "syn" if img.startswith("syn_") else ("rot" if "_r" in img and img.split("_r")[-1] in ("090", "180", "270") else "real")
            rows.append(make_row(img, prompt, main_answer(r, cls), "main", src))
        return rows

    # ── S1 原子段 ──
    atomic_imgs_realrot = perm[:2500]
    atomic_imgs = atomic_imgs_realrot + sorted(syn.keys())
    assert len(atomic_imgs) == SEG_ATOMIC
    rngT = random.Random(3407)
    atomic_rows = []
    template_used = Counter()
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        want = TEMPLATE_COUNTS[tid]
        for img in draw_rounds(atomic_imgs, want, rngT):
            r, _ = geom_of(img)
            src = "syn" if img.startswith("syn_") else ("rot" if "_r" in img else "real")
            atomic_rows.append(make_row(img, templates[tid], atomic_answer(r, tid), "atomic", src))
            template_used[tid] += 1
    eligible6 = [img for img in atomic_imgs if geom_of(img)[0]["angular_type"] in ("single", "opposed")]
    assert len(eligible6) > 0
    for img in draw_rounds(eligible6, TEMPLATE_COUNTS["T6"], rngT):
        r, _ = geom_of(img)
        src = "syn" if img.startswith("syn_") else ("rot" if "_r" in img else "real")
        atomic_rows.append(make_row(img, templates["T6"], atomic_answer(r, "T6"), "atomic", src))
        template_used["T6"] += 1
    random.Random(3407).shuffle(atomic_rows)

    # ── S4 原子段（只用原图） ──
    rngS4 = random.Random(3407)
    atomic_rows_s4 = []
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        for img in draw_rounds(real_names, TEMPLATE_COUNTS[tid], rngS4):
            r, _ = geom_of(img)
            atomic_rows_s4.append(make_row(img, templates[tid], atomic_answer(r, tid), "atomic", "real"))
    eligible6_s4 = [img for img in real_names if geo[img]["angular_type"] in ("single", "opposed")]
    assert len(eligible6_s4) > 0
    for img in draw_rounds(eligible6_s4, TEMPLATE_COUNTS["T6"], rngS4):
        atomic_rows_s4.append(make_row(img, templates["T6"], atomic_answer(geo[img], "T6"), "atomic", "real"))
    random.Random(3407).shuffle(atomic_rows_s4)

    # ── 组装并写出 ──
    s1_rows = atomic_rows + build_main_rows(s1_main)
    s4_rows = atomic_rows_s4 + build_main_rows(s4_main)
    s0_rows = build_main_rows(s0_main)
    assert len(s1_rows) == len(s4_rows) == len(s0_rows) == N_MAIN

    def write_rows(name, rows):
        p = HERE / name
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        return sha

    sha_s1 = write_rows("sft_S1.jsonl", s1_rows)
    shutil.copyfile(HERE / "sft_S1.jsonl", HERE / "sft_S2.jsonl")
    shutil.copyfile(HERE / "sft_S1.jsonl", HERE / "sft_S1p.jsonl")
    # S3：与 S1 相同，仅图片名加 _x2
    with open(HERE / "sft_S1.jsonl", encoding="utf-8") as fin, \
         open(HERE / "sft_S3.jsonl", "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            obj = json.loads(line)
            obj["images"][0] = obj["images"][0].replace(".png", "_x2.png")
            fout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sha_s0 = write_rows("sft_S0.jsonl", s0_rows)
    sha_s4 = write_rows("sft_S4.jsonl", s4_rows)
    shas = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
            for n in ("sft_S0.jsonl", "sft_S1.jsonl", "sft_S2.jsonl", "sft_S3.jsonl",
                      "sft_S4.jsonl", "sft_S1p.jsonl")}

    # ── 校验 ──
    checks = []
    checks.append(("六文件均 12000 行",
                   all(sum(1 for _ in open(HERE / n, encoding="utf-8")) == N_MAIN for n in shas)))
    checks.append(("S1=S2=S1p 逐字节", len({shas['sft_S1.jsonl'], shas['sft_S2.jsonl'], shas['sft_S1p.jsonl']}) == 1))
    # S3 与 S1 仅 images 差 _x2
    same = True
    with open(HERE / "sft_S1.jsonl", encoding="utf-8") as f1, open(HERE / "sft_S3.jsonl", encoding="utf-8") as f3:
        for l1, l3 in zip(f1, f3):
            o1, o3 = json.loads(l1), json.loads(l3)
            if {k: v for k, v in o1.items() if k != "images"} != {k: v for k, v in o3.items() if k != "images"}:
                same = False; break
            if o3["images"][0] != o1["images"][0].replace(".png", "_x2.png"):
                same = False; break
    checks.append(("S3 与 S1 仅 images 字段差 _x2", same))

    # ── 200 条主任务重算 ──
    idx = random.Random(3407).sample(range(N_MAIN), 200)
    n_ok = 0
    for k in idx:
        row = s0_rows[k]
        img = row["sample_id"]
        if "_r" in img and img.split("_r")[-1] in ("090", "180", "270"):
            src = img.split("_r")[0]
            kk = {"090": -1, "180": -2, "270": -3}[img.split("_r")[-1]]
            m = np.rot90(np.load(HERE / "matrices" / f"{src}.npy"), k=kk)
            cls = geo[src]["failure_type"]
        else:
            m = np.load(HERE / "matrices" / f"{img}.npy")
            cls = geo[img]["failure_type"]
        g = core.measure_v3(m)
        ref = core.reference(g, cls)
        exp1 = measure_line({"coverage": g["coverage"], "zone_frac": [g["radial"]["zone_frac"][x] for x in ("center", "middle", "edge")],
                             "R1": g["direction"].get("R1_concentration", 0.0), "R2": g["direction"].get("R2_axial", 0.0)})
        exp = exp1 + "\n" + json.dumps(ref, ensure_ascii=False, separators=(",", ":"))
        if exp == row["messages"][1]["content"]:
            n_ok += 1
    checks.append((f"200 条主任务重算一致 {n_ok}/200", n_ok == 200))
    # 受保护/池外检查
    all_src = set()
    for rows in (s0_rows, s1_rows, s4_rows):
        for r in rows:
            img = r["sample_id"]
            all_src.add(img.split("_r")[0] if img.split("_r")[-1] in ("090", "180", "270") else img)
    out_of_pool = {s for s in all_src if not s.startswith("syn_") and s not in pool_set}
    checks.append(("训练图全部属于 TRAIN_POOL 或合成", not out_of_pool))

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if not all(ok for _, ok in checks):
        print("**验收不过，按 0.4 决策表处理**")

    # ── 统计 ──
    def file_stats(rows, atomic_tmpl=None):
        st = {"lines": len(rows), "atomic": 0, "main": 0, "line_src": Counter(), "main_line_src": Counter(),
              "main_angular": Counter(), "template": Counter()}
        tmpl_by_text = {v: k for k, v in templates.items()}
        for r in rows:
            st[r["task"]] += 1
            st["line_src"][r["line_src"]] += 1
            if r["task"] == "main":
                st["main_line_src"][r["line_src"]] += 1
                img = r["sample_id"]
                rr, _ = geom_of(img)
                st["main_angular"][rr["angular_type"]] += 1
            else:
                txt = r["messages"][0]["content"][len("<image>\n"):]
                st["template"][tmpl_by_text[txt]] += 1
        return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}

    stats = {"per_file": {}, "atomic_images_S1": {"realrot": len(atomic_imgs_realrot),
                                                  "syn": len(atomic_imgs) - len(atomic_imgs_realrot)},
             "template_counts_S1": dict(template_used),
             "shas": shas,
             "main_draw": {"s0_s1_prefix": "同一 3407 排列，S0 取前 12000、S1 取前 8000",
                           "s4_mode": "原图池按轮次重复抽取"}}
    for n, rows in (("sft_S0", s0_rows), ("sft_S1", s1_rows), ("sft_S4", s4_rows)):
        stats["per_file"][n] = file_stats(rows)
        print(f"{n}: {json.dumps(stats['per_file'][n], ensure_ascii=False)}")
    (HERE / "sft_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")

    # 数据清单
    manifest = HERE / "数据清单.json"
    items = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    items = [it for it in items if it["file"] not in shas]
    for n, sha in shas.items():
        items.append({"file": n, "lines": N_MAIN, "sha256": sha})
    manifest.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    # 抽样打印
    print("\n===== 主任务样例 =====")
    for r in s0_rows[:3]:
        print(json.dumps(r, ensure_ascii=False)[:400])
    print("\n===== 原子样例 =====")
    for r in s1_rows[:2] + s1_rows[2000:2001]:
        print(json.dumps(r, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
