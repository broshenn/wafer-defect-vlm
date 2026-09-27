"""Trivial baselines for the structured and caption tasks.

The benchmark shipped floors for classification and retrieval but not for the
structured fields, and that omission hid the single most important number in the
whole evaluation. `radial_zone` is 83% "center", so answering "center" every
time scores 0.833 — against which the fine-tuned model's 0.264 is not a modest
result but a failure. An accuracy quoted without its majority baseline invites
exactly that misreading, so the floors are computed here for every field.

Two floors per field, because they answer different questions:
  * majority / best-constant — what a model that ignores the image scores.
    This is the bar a real model must clear.
  * uniform random — what a model that guesses among the observed values scores.
    This is the bar for "has any signal at all".
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def circular_mae(prediction: int, truth: int, period: int = 12) -> int:
    """Distance on a circle: 12 and 1 are one step apart, not eleven."""
    delta = abs(int(prediction) - int(truth)) % period
    return min(delta, period - delta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    bench = Path(args.benchmark)
    rows = [json.loads(line) for line in (bench / "structured.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    golds = [row["gold"] for row in rows]
    result: dict[str, object] = {"samples": len(rows), "note": (
        "best_constant is what a model that ignores the image scores, and is the "
        "bar that matters. uniform_random is the bar for having any signal.")}

    # ---------------------------------------------------------- categorical
    for field in ("defect_type", "radial_zone"):
        values = [g[field] for g in golds if g.get(field) is not None]
        counts = collections.Counter(values)
        top, top_count = counts.most_common(1)[0]
        k = len(counts)
        result[field] = {
            "n": len(values),
            "values": dict(counts),
            "cardinality": k,
            "best_constant": top,
            "best_constant_accuracy": top_count / len(values),
            "uniform_random_accuracy": 1 / k,
        }

    # --------------------------------------------------------------- circular
    sectors = [g["clock_sector"] for g in golds if g.get("clock_sector") is not None]
    if sectors:
        # The best constant under a circular metric is the point minimising the
        # summed circular distance — not the arithmetic mean, which is
        # meaningless on a clock face.
        candidates = sorted(set(sectors) | set(range(1, 13)))
        best = min(candidates, key=lambda c: sum(circular_mae(c, s) for s in sectors))
        best_mae = statistics.mean(circular_mae(best, s) for s in sectors)
        random_mae = statistics.mean(
            circular_mae(a, b) for a in candidates for b in sectors)
        result["clock_sector"] = {
            "n": len(sectors),
            "best_constant": best,
            "best_constant_circular_mae": best_mae,
            "uniform_random_circular_mae": random_mae,
        }

    # --------------------------------------------------------------- numeric
    sizes = [g["size_r"] for g in golds if g.get("size_r") is not None]
    if sizes:
        median = statistics.median(sizes)
        mean = statistics.fmean(sizes)
        result["size_r"] = {
            "n": len(sizes),
            "best_constant": median,
            "best_constant_mae": statistics.fmean(abs(s - median) for s in sizes),
            "mean_constant_mae": statistics.fmean(abs(s - mean) for s in sizes),
            "range": [min(sizes), max(sizes)],
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items() if k not in ("note",)}, indent=2,
                     ensure_ascii=False))

    if args.markdown:
        lines = ["# 结构化字段的平凡基线 / Trivial baselines, structured fields", ""]
        for field in ("defect_type", "radial_zone"):
            entry = result.get(field)
            if entry:
                lines += [f"## {field}", "",
                          f"- 取值分布 {entry['values']}",
                          f"- 恒答多数类的准确率 **{entry['best_constant_accuracy']:.4f}**"
                          f"（答案恒为 `{entry['best_constant']}`）",
                          f"- 均匀随机准确率 {entry['uniform_random_accuracy']:.4f}（k={entry['cardinality']}）",
                          ""]
        for field in ("clock_sector", "size_r"):
            entry = result.get(field)
            if entry:
                mae_key = next(k for k in entry if k.endswith("_mae") and "constant" in k)
                lines += [f"## {field}", "",
                          f"- 恒答 `{entry['best_constant']}` 的 MAE **{entry[mae_key]:.4f}**"
                          f"（这是「忽略图像」的下限）", ""]
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
