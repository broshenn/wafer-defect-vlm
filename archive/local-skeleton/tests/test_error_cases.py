"""Tests for the error-case extractor.

Two things are worth pinning down here. The first is the alignment trap:
swift infer writes no sample_id into its result rows, so pairing is positional
and only correct against the full request list. The second is honesty about
confidence — a run without log-probabilities must never yield a case described
as high-confidence.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


def _find_tool(name: str) -> Path:
    for base in Path(__file__).resolve().parents:
        candidate = base / "tools" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _find_tool(name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cases = _load("error_cases.py")


def _entry(token: str, logprob: float) -> dict:
    return {"token": token, "logprob": logprob, "bytes": list(token.encode("utf8"))}


def test_confidence_ignores_the_thinking_block() -> None:
    # The thinking tokens are the model talking to itself; only the answer is
    # being scored, and a confident guess buried after an uncertain preamble
    # should still read as confident.
    entries = [
        _entry("<think>", -8.0), _entry("hmm", -7.0), _entry("</think>", -0.01),
        _entry("Center", -0.02),
    ]
    assert cases.confidence_of(entries) == pytest.approx(math.exp(-0.02), rel=1e-6)


def test_answer_span_is_everything_when_there_is_no_thinking_block() -> None:
    entries = [_entry("Donut", -0.5), _entry("!", -1.0)]
    assert cases.answer_span(entries) == entries
    assert cases.confidence_of(entries) == pytest.approx(math.exp(-0.75), rel=1e-6)


def test_unmeasured_confidence_is_none_not_zero() -> None:
    # The distinction the whole report depends on.
    assert cases.confidence_of([]) is None
    assert cases.confidence_of([_entry("Center", None)]) is None
    assert cases.flatten_logprobs(None) == []
    assert cases.flatten_logprobs({"content": [_entry("x", -1.0)]})[0]["token"] == "x"


def test_multibyte_tokens_are_measured_in_bytes() -> None:
    # Offsets come from the recorded bytes because decoding a token alone is not
    # always concatenation-safe for CJK.
    entries = [_entry("<think>", -1.0), _entry("思考", -2.0), _entry("</think>", -1.0),
               _entry("边缘", -0.25)]
    span = cases.answer_span(entries)
    assert [e["token"] for e in span] == ["边缘"]


def _write_benchmark(root: Path, n_other: int) -> None:
    """A miniature benchmark where a non-classification task comes first."""
    root.mkdir(parents=True, exist_ok=True)
    requests = [{"sample_id": f"other{i}", "task": "caption"} for i in range(n_other)]
    gold = ["Center", "Donut", "Scratch", "none"]
    for i, label in enumerate(gold):
        requests.append({"sample_id": f"cls{i}", "task": "classification"})
    with (root / "all_requests.jsonl").open("w", encoding="utf-8") as fh:
        for row in requests:
            fh.write(json.dumps(row) + "\n")
    with (root / "classification.jsonl").open("w", encoding="utf-8") as fh:
        for i, label in enumerate(gold):
            fh.write(json.dumps({"sample_id": f"cls{i}", "gold_label": label}) + "\n")


def test_alignment_must_happen_before_filtering(tmp_path) -> None:
    # Two of four correct. If the tool filtered to classification before
    # aligning, the classification rows would be paired against the caption
    # predictions at the front of the file and the score would collapse.
    benchmark = tmp_path / "bench"
    _write_benchmark(benchmark, n_other=3)
    responses = ["a caption"] * 3 + ["Center", "wrong", "Scratch", "wrong"]
    predictions = tmp_path / "pred.jsonl"
    with predictions.open("w", encoding="utf-8") as fh:
        for text in responses:
            fh.write(json.dumps({"response": text, "logprobs": None}) + "\n")

    output = tmp_path / "ec.json"
    argv = sys.argv
    sys.argv = ["error_cases.py", "--benchmark", str(benchmark),
                "--predictions", str(predictions), "--output", str(output)]
    try:
        assert cases.main() == 0
    finally:
        sys.argv = argv

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["total"] == 4
    assert report["correct"] == 2
    assert report["accuracy"] == 0.5


def test_logprobs_are_read_from_the_positional_prediction_row(tmp_path) -> None:
    # swift infer rows carry no sample_id, so the log-probabilities can only be
    # reached positionally. A by-sample_id lookup finds nothing and quietly
    # reports every confidence as unmeasured.
    benchmark = tmp_path / "bench"
    _write_benchmark(benchmark, n_other=0)
    # Gold order is Center, Donut, Scratch, none.
    predictions = tmp_path / "pred.jsonl"
    with predictions.open("w", encoding="utf-8") as fh:
        for text, logprob in (("Center", -0.01), ("Scratch", -0.001),
                              ("Scratch", -0.02), ("none", -0.5)):
            entries = [_entry("<think>", -3.0), _entry("</think>", -0.1), _entry(text, logprob)]
            fh.write(json.dumps({"response": f"<think></think>{text}",
                                 "logprobs": {"content": entries}}) + "\n")

    output = tmp_path / "ec.json"
    argv = sys.argv
    sys.argv = ["error_cases.py", "--benchmark", str(benchmark),
                "--predictions", str(predictions), "--output", str(output)]
    try:
        cases.main()
    finally:
        sys.argv = argv

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["confidence_is_measured"] is True
    assert report["confidence_measured_for"] == 4
    # Row 2 answered "Scratch" where the gold is "Donut", and did so with the
    # highest confidence of the four, so it must head the error list.
    assert report["high_confidence_errors"][0]["predicted"] == "Scratch"
    assert report["high_confidence_errors"][0]["gold"] == "Donut"
    assert len(report["high_confidence_errors"]) == 1


def test_no_logprobs_means_no_high_confidence_claim(tmp_path) -> None:
    benchmark = tmp_path / "bench"
    _write_benchmark(benchmark, n_other=0)
    predictions = tmp_path / "pred.jsonl"
    with predictions.open("w", encoding="utf-8") as fh:
        for text in ["Center", "Donut", "Scratch", "none"]:
            fh.write(json.dumps({"response": text, "logprobs": None}) + "\n")

    output = tmp_path / "ec.json"
    argv = sys.argv
    sys.argv = ["error_cases.py", "--benchmark", str(benchmark),
                "--predictions", str(predictions), "--output", str(output)]
    try:
        cases.main()
    finally:
        sys.argv = argv

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["confidence_is_measured"] is False
    assert report["high_confidence_errors"] == []
    assert "no case can be called high-confidence" in report["confidence_note"]
