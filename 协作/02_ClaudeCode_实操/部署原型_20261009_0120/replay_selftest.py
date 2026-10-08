"""对实际HTTP服务回放冻结36图；只测工程/类别，描述内容另作模型自查。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full", "Random", "Scratch", "none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--samples", required=True, help="冻结36图样本清单.json")
    ap.add_argument("--images", required=True, help="36图PNG目录")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logpath = out / "http_replay.jsonl"
    if logpath.exists():
        raise SystemExit("拒绝重复回放覆盖结果")
    manifest = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = manifest["样本"]
    assert len(samples) == 36 and len({r["sample_id"] for r in samples}) == 36
    assert len({r["图号"] for r in samples}) == 36
    for s in samples:
        image = Path(args.images) / (s["sample_id"] + ".png")
        assert hashlib.sha256(image.read_bytes()).hexdigest() == s["图sha256"]
    with urllib.request.urlopen(args.url + "/health", timeout=15) as r:
        health = json.load(r)
    assert health["status"] == "ready"
    stats, rows = {}, []
    with logpath.open("x", encoding="utf-8") as log:
        for sample in samples:
            image = Path(args.images) / (sample["sample_id"] + ".png")
            t0 = time.perf_counter()
            req = urllib.request.Request(args.url + "/predict", data=image.read_bytes(), headers={"Content-Type": "image/png"})
            status, result = 0, None
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    status, result = response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                status = exc.code
                result = json.loads(exc.read().decode("utf-8"))
            except Exception as exc:
                result = {"error_type": type(exc).__name__}
            item = {"number": sample["图号"], "sample_id": sample["sample_id"],
                    "gold_class": sample["公开类别"], "http_status": status,
                    "client_seconds": time.perf_counter() - t0, "response": result}
            rows.append(item)
            log.write(json.dumps(item, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"number": sample["图号"], "http_status": status,
                              "client_seconds": round(item["client_seconds"], 3)}, ensure_ascii=False), flush=True)
    successful = [r for r in rows if r["http_status"] == 200]
    usable = [r for r in successful if r["response"].get("result") is not None]
    times = sorted(r["client_seconds"] for r in successful)
    get_pred = lambda r: (r["response"].get("result") or {}).get("defect_class")
    macro = []
    for category in CLASSES:
        tp = sum(r["gold_class"] == category and get_pred(r) == category for r in rows)
        fp = sum(r["gold_class"] != category and get_pred(r) == category for r in rows)
        fn = sum(r["gold_class"] == category and get_pred(r) != category for r in rows)
        macro.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    stats = {"scope": "冻结36图开发回放；不是新盲测，非人工gold，不把功能通过当描述事实通过", "n": 36,
             "health": health, "http_200": len(successful), "schema_result_available": len(usable),
             "needs_review": sum(bool(r["response"].get("needs_review")) for r in successful),
             "correct_class": sum(get_pred(r) == r["gold_class"] for r in rows),
             "macro_f1_fixed9": sum(macro) / 9,
             "latency_scope": "同客户端HTTP提交至完整响应；首条单列，全部是实际请求",
             "first_request_seconds": rows[0]["client_seconds"],
             "p50_seconds_all": statistics.median(times) if times else None,
             "p95_seconds_all": times[min(len(times) - 1, math.ceil(len(times) * 0.95) - 1)] if times else None,
             "max_peak_allocated_gib": max((r["response"].get("peak_allocated_gib", 0) for r in successful), default=None),
             "morphology_position_direction_fact_accuracy": "本脚本未测，需对真实原图和实际response做逐图模型自查"}
    (out / "selftest_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if len(successful) == 36 and len(usable) >= 35 else 1


if __name__ == "__main__":
    raise SystemExit(main())
