#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 图池筛选（B 卡 10；合同 6.5）。
分片：python screen_rl_pool.py --shard i --nshards 6 --winner winner.json --model <基座> \
      --candidates rl_candidates800.jsonl --out-dir night/screen --batch-size 8
      每图采样 4 次（temperature 1.0，max_tokens 384，seed 3407+k），用 full 奖励打分。
合并：python screen_rl_pool.py --merge --out-dir ... --template rsft_rows_template.jsonl \
      --out-pool rl_pool320.jsonl --out-rsft rsft_pool320.jsonl --passk passk.json
      组内奖励不全相同的图入选；按候选顺序取前 160；不足按顺序补足并记录；pass@1/pass@4 写入 passk.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_v5 import parse_raw_strict, parse_v5  # noqa: E402
from reward_v5 import score_v5  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def geo_correct(text, ref) -> bool:
    """几何三字段全对算对（参考 unknown 的字段不参评）。"""
    p = parse_v5(text)
    if not p["ok"]:
        return False
    for field in ("coverage_band", "red_mass_zones", "angular_type"):
        if ref[field] != "unknown" and p["obj"][field] != ref[field]:
            return False
    return True


def merge(rows_by_shard, candidates, template_rows, n_select=160, n_reserve=320):
    """合并逻辑（纯函数）。返回 (selected, passk, n_filled_from_remainder)。"""
    by_sid = {}
    for shard in rows_by_shard:
        for r in shard:
            by_sid.setdefault(r["sample_id"], []).extend(r["samples"])
    scored = []
    for cand in candidates:
        sid = cand["sample_id"]
        ref = parse_raw_strict(cand["reference_json"])
        samples = by_sid.get(sid, [])
        rewards = [s["reward"] for s in samples]
        oks = [s["format_ok"] for s in samples]
        geo = [geo_correct(s["raw"], ref) for s in samples]
        varying = len(set(rewards)) > 1 if rewards else False
        scored.append({"sample_id": sid, "rewards": rewards, "format_ok": oks,
                       "geo_correct": geo, "varying": varying, "n": len(samples)})
    selected = [s for s in scored if s["varying"]][:n_select]
    n_fill = 0
    if len(selected) < n_select:
        chosen = {s["sample_id"] for s in selected}
        for s in scored:
            if len(selected) >= n_select:
                break
            if s["sample_id"] not in chosen:
                selected.append(s)
                n_fill += 1
    sel_ids = [s["sample_id"] for s in selected]
    n_pass1 = sum(1 for s in scored if s["geo_correct"] and s["geo_correct"][0])
    n_pass4 = sum(1 for s in scored if s["geo_correct"] and any(s["geo_correct"]))
    passk = {"n_candidates": len(scored), "n_selected": len(selected),
             "n_filled_from_remainder": n_fill,
             "pass@1": round(n_pass1 / len(scored), 4) if scored else None,
             "pass@4": round(n_pass4 / len(scored), 4) if scored else None,
             "zero_variance": sum(1 for s in scored if not s["varying"]),
             "note": "pass 定义：几何三字段全对（coverage_band/red_mass_zones/angular_type，unknown 字段不参评）"}
    return sel_ids, passk, scored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--winner", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--template", default=None)
    ap.add_argument("--out-pool", default=None)
    ap.add_argument("--out-rsft", default=None)
    ap.add_argument("--passk", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.merge:
        shards = [load_jsonl(out_dir / f"screen_shard_{i}.jsonl") for i in range(a.nshards)]
        candidates = load_jsonl(a.candidates) if a.candidates else sum(shards, [])
        # merge 模式可用 --candidates 指定原顺序；否则用分片并集（顺序按 shard）
        cand_rows = candidates
        template = load_jsonl(a.template)
        tmpl_by_sid = {}
        for r in template:
            tmpl_by_sid.setdefault(r["sample_id"], []).append(r)
        sel_ids, passk, scored = merge(shards, cand_rows, template)
        pool_rows, rsft_rows = [], []
        cand_by_sid = {c["sample_id"]: c for c in cand_rows}
        for sid in sel_ids:
            c = cand_by_sid[sid]
            pool_rows.append({"messages": c["messages"], "images": c["images"],
                              "sample_id": sid, "reference_json": c["reference_json"]})
            rsft_rows.extend(tmpl_by_sid[sid])
        with open(a.out_pool, "w", encoding="utf-8", newline="\n") as f:
            for r in pool_rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        with open(a.out_rsft, "w", encoding="utf-8", newline="\n") as f:
            for r in rsft_rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        Path(a.passk).write_text(json.dumps(passk, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8", newline="\n")
        with open(out_dir / "screen_scored.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"merge：池 {len(pool_rows)}、rsft {len(rsft_rows)}，补足 {passk['n_filled_from_remainder']}；"
              f"pass@1={passk['pass@1']} pass@4={passk['pass@4']}")
        return 0

    # 分片模式
    assert a.shard is not None and a.winner and a.model and a.candidates
    os.environ["WAFER_REWARD_MODE"] = "full"
    winner = json.loads(Path(a.winner).read_text(encoding="utf-8"))
    rows = load_jsonl(a.candidates)
    mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    print(f"分片 {a.shard}/{a.nshards}：{len(mine)} 张")

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父 shell 未锁 GPU"
    import torch
    torch.cuda.init()
    assert torch.cuda.device_count() == 1, "必须真单卡"
    from swift import InferRequest, RequestConfig
    from swift.infer_engine import TransformersEngine
    t0 = time.perf_counter()
    eng = TransformersEngine(a.model, model_type="qwen3_5", template_type="qwen3_5",
                             adapters=[winner["checkpoint"]])
    print(f"加载 {time.perf_counter() - t0:.1f}s 起点 {winner['checkpoint']}")

    out_path = out_dir / f"screen_shard_{a.shard}.jsonl"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for k, r in enumerate(mine, 1):
            ref = parse_raw_strict(r["reference_json"])
            samples = []
            for g in range(4):
                cfg = RequestConfig(max_tokens=384, temperature=1.0, seed=3407 + g)
                try:
                    resp = eng.infer([InferRequest(messages=r["messages"],
                                                   images=r["images"])], cfg)[0]
                    raw = resp.choices[0].message.content
                except Exception as e:
                    raw = f"<infer_error {type(e).__name__}>"
                sc = score_v5(raw, ref, "full")
                samples.append({"gen": g, "seed": 3407 + g, "raw": raw,
                                "reward": sc["reward"], "format_ok": sc["ok"]})
            f.write(json.dumps({"sample_id": r["sample_id"], "samples": samples},
                               ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"  [{k}/{len(mine)}] {r['sample_id']}", flush=True)
    print(f"分片 {a.shard} 完成 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
