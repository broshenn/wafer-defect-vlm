"""Extract success, failure and high-confidence-error cases from a scored run.

The interesting cases are not the average ones. A model that is wrong while
hedging is making a different kind of mistake from one that is wrong and
certain, and only the second tells you the training signal is confidently
mislabelled somewhere.

Confidence comes from the token log-probabilities ms-swift records when
inference is run with ``--logprobs true``. It is the geometric mean of the
answer tokens' probabilities, i.e. ``exp(mean(logprob))`` over everything the
model generated after its thinking block — a per-token average, not a product,
so a long answer is not penalised just for being long.

When log-probabilities were not requested the confidence is ``None`` and the
case is never described as high-confidence. An unmeasured confidence and a low
one are different findings, and conflating them would invent evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.evaluate import align_predictions, predicted_label, read_jsonl  # noqa: E402

THINK_END = "</think>"


def answer_span(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the tokens generated after the thinking block.

    Offsets are computed from the recorded UTF-8 bytes rather than by decoding
    each token in isolation, because a per-token decode is not always
    concatenation-safe for multi-byte characters.
    """
    if not entries:
        return []
    text = b"".join(bytes(e.get("bytes") or []) for e in entries).decode("utf-8", "replace")
    marker = text.rfind(THINK_END)
    if marker < 0:
        return entries
    cutoff = len(text[: marker + len(THINK_END)].encode("utf-8"))

    span, offset = [], 0
    for entry in entries:
        size = len(bytes(entry.get("bytes") or []))
        if offset >= cutoff:
            span.append(entry)
        offset += size
    return span or entries


def confidence_of(entries: list[dict[str, Any]]) -> float | None:
    """Geometric mean of the answer tokens' probabilities, or None if unmeasured."""
    span = answer_span(entries)
    values = [float(e["logprob"]) for e in span if e.get("logprob") is not None]
    if not values:
        return None
    return math.exp(sum(values) / len(values))


def flatten_logprobs(raw: Any) -> list[dict[str, Any]]:
    """ms-swift stores {'content': [...]}; accept that or a bare list."""
    if isinstance(raw, dict):
        return list(raw.get("content") or [])
    if isinstance(raw, list):
        return list(raw)
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, help="benchmark directory")
    ap.add_argument("--predictions", required=True, help="result jsonl from swift infer")
    ap.add_argument("--requests", default=None,
                    help="request rows to align against; defaults to the classification split")
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    bench = Path(args.benchmark)
    # The requests must be the same rows inference actually saw, which is
    # all_requests.jsonl: classification.jsonl carries the gold label but no
    # `task` field, so aligning against it silently falls through to positional
    # pairing and every case comes back wrong. The gold map is keyed by
    # sample_id for exactly this reason.
    requests_path = Path(args.requests) if args.requests else bench / "all_requests.jsonl"
    all_requests = read_jsonl(requests_path)
    gold_map = {r["sample_id"]: r["gold_label"] for r in read_jsonl(bench / "classification.jsonl")}
    raw_predictions = read_jsonl(args.predictions)
    # swift infer does not write sample_id back into its result rows, so
    # align_predictions pairs by position. That is only sound against the full
    # request list in its original order, which is how the evaluator calls it;
    # filtering to one task first would pair classification rows against
    # whatever happened to sit at that index. Align in full, then filter.
    aligned = align_predictions(all_requests, raw_predictions)

    # The same resolution rule align_predictions applies, repeated here because
    # the log-probabilities live on the raw prediction row and align_predictions
    # only returns the response text. swift infer writes no sample_id into its
    # result rows, so in practice this is positional — and looking the row up by
    # sample_id instead silently returns nothing, which is how the first run
    # reported every confidence as unmeasured while the logprobs sat right there
    # in the file.
    keyed = {
        (str(row.get("sample_id")), str(row.get("task"))): row
        for row in raw_predictions
        if row.get("sample_id") is not None and row.get("task") is not None
    }

    records = []
    for index, row in enumerate(aligned):
        if row.get("task") != "classification":
            continue
        raw = keyed.get((str(row.get("sample_id")), str(row.get("task"))))
        if raw is None and index < len(raw_predictions):
            raw = raw_predictions[index]
        raw = raw or {}
        gold = gold_map.get(row.get("sample_id"))
        guess = predicted_label(row.get("response", ""))
        records.append({
            "sample_id": row.get("sample_id"),
            "task": row.get("task"),
            "lot_name": row.get("lot_name"),
            "image_path": row.get("image_path"),
            "gold": gold,
            "predicted": guess,
            "correct": guess == gold,
            "confidence": confidence_of(flatten_logprobs(raw.get("logprobs"))),
            "response": row.get("response", ""),
        })

    measured = [r for r in records if r["confidence"] is not None]
    correct = [r for r in records if r["correct"]]
    wrong = [r for r in records if not r["correct"]]

    ranked_wrong = sorted(
        [r for r in wrong if r["confidence"] is not None],
        key=lambda r: r["confidence"], reverse=True)
    ranked_right = sorted(
        [r for r in correct if r["confidence"] is not None],
        key=lambda r: r["confidence"], reverse=True)

    report = {
        "task": "classification",
        "requests": str(requests_path),
        "predictions": args.predictions,
        "total": len(records),
        "correct": len(correct),
        "incorrect": len(wrong),
        "accuracy": (len(correct) / len(records)) if records else None,
        "confidence_measured_for": len(measured),
        "confidence_definition": (
            "exp(mean(token logprob)) over the tokens generated after the thinking "
            "block; per-token average, so answer length does not penalise the score."
        ),
        "confidence_is_measured": len(measured) > 0,
        # Only ever populated from measured confidences. Empty means "not
        # measured", which the reader must not read as "no confident errors".
        "high_confidence_errors": ranked_wrong[: args.top_n],
        "confident_successes": ranked_right[: args.top_n],
        "all_failures": ranked_wrong[: 200],
    }
    if not measured:
        report["confidence_note"] = (
            "No log-probabilities were present in this run, so no case can be "
            "called high-confidence. Re-run inference with --logprobs true."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.markdown:
        lines = [f"# Error cases — {report['task']}", ""]
        lines.append(f"- total {report['total']}, correct {report['correct']}, "
                     f"incorrect {report['incorrect']}")
        lines.append(f"- confidence measured for {len(measured)}/{len(records)}")
        if not measured:
            lines.append(f"- **{report['confidence_note']}**")
        lines += ["", f"## Most confident errors (top {args.top_n})", "",
                  "| sample | gold | predicted | confidence | lot |", "|---|---|---|---|---|"]
        for r in report["high_confidence_errors"]:
            c = "n/a" if r["confidence"] is None else f"{r['confidence']:.3f}"
            lines.append(f"| {r['sample_id']} | {r['gold']} | {r['predicted']} | {c} | {r['lot_name']} |")
        lines += ["", f"## Most confident successes (top {args.top_n})", "",
                  "| sample | gold | predicted | confidence | lot |", "|---|---|---|---|---|"]
        for r in report["confident_successes"]:
            c = "n/a" if r["confidence"] is None else f"{r['confidence']:.3f}"
            lines.append(f"| {r['sample_id']} | {r['gold']} | {r['predicted']} | {c} | {r['lot_name']} |")
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "total": report["total"], "correct": report["correct"],
        "incorrect": report["incorrect"], "accuracy": report["accuracy"],
        "confidence_measured_for": len(measured),
        "high_confidence_errors": len(report["high_confidence_errors"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
