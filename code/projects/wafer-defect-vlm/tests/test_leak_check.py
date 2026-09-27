"""Tests for the leakage checker.

The subtlety worth pinning: not every overlap is a defect. GRPO is an RL pass
over the SFT training split, so it is supposed to reuse those wafers, while
nothing may ever touch the benchmark or the validation split.
"""

import importlib.util
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "projects" / "wafer-defect-vlm" / "src"
sys.path.insert(0, str(SRC))

from wafer_vlm.utils import split_for_lot  # noqa: E402

SEED = 3407


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


leak = _load("leak_check.py")


def _lots_for(split: str, count: int) -> list[str]:
    """Find real lot names that hash into the requested split."""
    found = []
    i = 0
    while len(found) < count:
        lot = f"lot{i}"
        if split_for_lot(lot, SEED) == split:
            found.append(lot)
        i += 1
    return found


def _build(tmp_path: Path) -> dict:
    """A clean, fully isolated layout."""
    bench = tmp_path / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    curated = tmp_path / "curated"
    (curated / "splits").mkdir(parents=True, exist_ok=True)

    train_lots, val_lots, test_lots = (_lots_for("train", 2), _lots_for("val", 1),
                                       _lots_for("test", 2))
    manifest, train_rows, core_rows = [], [], []
    for split, lots in (("train", train_lots), ("val", val_lots), ("test", test_lots)):
        for lot in lots:
            sample_id = f"{lot}_000"
            manifest.append({"sample_id": sample_id, "lot_name": lot, "split": split})
            if split == "test":
                core_rows.append({"sample_id": sample_id, "lot_name": lot, "split": split})
            elif split == "train":
                train_rows.append({"sample_id": sample_id, "task": "classification"})

    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest) + "\n", encoding="utf-8")
    (bench / "core.jsonl").write_text(
        "\n".join(json.dumps(r) for r in core_rows) + "\n", encoding="utf-8")
    (curated / "splits" / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
    return {"bench": bench, "curated": curated, "manifest": tmp_path / "manifest.jsonl",
            "train_rows": train_rows, "test_rows": core_rows}


def _run(tmp_path: Path, layout: dict, grpo: Path | None = None) -> dict:
    output = tmp_path / "leak.json"
    argv = sys.argv
    command = ["leak_check.py", "--benchmark", str(layout["bench"]),
               "--curated", str(layout["curated"]), "--manifest", str(layout["manifest"]),
               "--output", str(output)]
    if grpo is not None:
        command += ["--grpo", str(grpo)]
    sys.argv = command
    try:
        leak.main()
    finally:
        sys.argv = argv
    return json.loads(output.read_text(encoding="utf-8"))


def test_isolated_layout_passes(tmp_path) -> None:
    report = _run(tmp_path, _build(tmp_path))
    assert report["passed"] is True
    assert report["failures"] == []
    assert report["rule_mismatch_count"] == 0
    assert report["benchmark"]["splits"] == ["test"]


def test_a_benchmark_sample_in_training_is_a_leak(tmp_path) -> None:
    layout = _build(tmp_path)
    # Move one test sample's id into the training file: the lot is then shared
    # with the benchmark even though the split column still says test.
    leaked = layout["test_rows"][0]["sample_id"]
    with (layout["curated"] / "splits" / "train.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sample_id": leaked, "task": "classification"}) + "\n")

    report = _run(tmp_path, layout)
    assert report["passed"] is False
    assert any("share" in f for f in report["failures"])
    assert report["sample_overlap_with_benchmark"]["train"]


def test_grpo_reusing_the_training_split_is_not_a_leak(tmp_path) -> None:
    # GRPO is RL on the SFT training split; flagging this would call the design
    # a defect. It is reported as an expected overlap instead.
    layout = _build(tmp_path)
    grpo = tmp_path / "grpo.jsonl"
    grpo.write_text("\n".join(json.dumps(r) for r in layout["train_rows"]) + "\n",
                    encoding="utf-8")

    report = _run(tmp_path, layout, grpo=grpo)
    assert report["passed"] is True
    assert "grpo/train" in report["expected_overlaps"]
    assert report["expected_overlaps"]["grpo/train"]["samples"] == len(layout["train_rows"])
    assert report["grpo_samples_outside_train_count"] == 0


def test_benchmark_drawn_from_a_non_test_split_is_a_leak(tmp_path) -> None:
    # The benchmark is meant to be test-split only; a train-split row inside it
    # would mean the headline number is partly measured on training data.
    layout = _build(tmp_path)
    train_sample = layout["train_rows"][0]["sample_id"]
    with (layout["bench"] / "core.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sample_id": train_sample, "lot_name": "x", "split": "train"}) + "\n")

    report = _run(tmp_path, layout)
    assert report["passed"] is False
    assert any("not drawn only from the test split" in f for f in report["failures"])
