import importlib.util
import json
from pathlib import Path


def _find_tool(name: str) -> Path:
    for base in Path(__file__).resolve().parents:
        candidate = base / "tools" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _find_tool(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report_tool = _load("make_report.py")
pick_tool = _load("pick_checkpoint.py")


RUN = {
    "classification": {"accuracy": 0.2143, "macro_f1": 0.1347,
                       "macro_f1_95ci": [0.0955, 0.1706], "invalid_predictions": 0},
    "structured": {"field_accuracy": {"defect_type": 0.21}, "clock_circular_mae_sectors": 5.0},
    "caption": {"must_hit_rate": 0.627},
    "robustness": {"accuracy": 0.1905, "flip_rate_vs_clean": 0.1389},
    "retrieval": None,
}


def test_dig_finds_nested_values() -> None:
    assert report_tool.dig(RUN, "classification.accuracy") == 0.2143
    assert report_tool.dig(RUN, "structured.field_accuracy.defect_type") == 0.21


def test_dig_returns_none_rather_than_raising_on_a_missing_path() -> None:
    # A whole run can be absent; the table must still render.
    assert report_tool.dig(RUN, "retrieval.mAP@10") is None
    assert report_tool.dig(RUN, "structured.nope.deeper") is None
    assert report_tool.dig({}, "classification.accuracy") is None


def test_missing_metric_is_not_rendered_as_zero() -> None:
    # The distinction this whole tool exists to preserve.
    assert report_tool.fmt(None) == "not run"
    assert report_tool.fmt(0.0) == "0.0000"


def test_confidence_interval_is_rendered_as_a_range() -> None:
    assert report_tool.fmt([0.0955, 0.1706]) == "[0.0955, 0.1706]"


def test_report_lists_which_metrics_were_never_measured(tmp_path) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(RUN), encoding="utf-8")
    output, markdown = tmp_path / "cmp.json", tmp_path / "cmp.md"
    main = report_tool.main
    import sys
    argv = sys.argv
    sys.argv = ["make_report.py", "--run", f"Base={base}",
                "--output", str(output), "--markdown", str(markdown)]
    try:
        assert main() == 0
    finally:
        sys.argv = argv
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "retrieval mAP@10" in report["not_measured"]["Base"]
    assert "classification accuracy" not in report["not_measured"]["Base"]
    text = markdown.read_text(encoding="utf-8")
    assert "| not run |" in text


def test_absent_run_file_is_reported_not_silently_dropped(tmp_path) -> None:
    output, markdown = tmp_path / "cmp.json", tmp_path / "cmp.md"
    import sys
    argv = sys.argv
    sys.argv = ["make_report.py", "--run", f"SFT={tmp_path / 'missing.json'}",
                "--output", str(output), "--markdown", str(markdown)]
    try:
        report_tool.main()
    finally:
        sys.argv = argv
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["runs"]["SFT"]["present"] is False
    assert "no result file" in markdown.read_text(encoding="utf-8")


def _write_state(run_dir: Path, name: str, step: int, losses: list[float]) -> Path:
    checkpoint = run_dir / name / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": step,
        "log_history": [{"eval_loss": loss} for loss in losses],
    }), encoding="utf-8")
    return checkpoint


def test_picks_the_lowest_eval_loss_checkpoint(tmp_path) -> None:
    _write_state(tmp_path, "v0", 100, [2.0])
    best = _write_state(tmp_path, "v0", 200, [1.5])
    _write_state(tmp_path, "v0", 300, [1.8])
    found = pick_tool.from_trainer_state(tmp_path)
    ranked = sorted(found, key=lambda item: item[1])
    assert ranked[0][0] == 200
    assert ranked[0][2] == best


def test_checkpoint_without_eval_loss_is_ignored(tmp_path) -> None:
    # Training can be stopped before the first eval; that must not win by default.
    _write_state(tmp_path, "v0", 100, [])
    _write_state(tmp_path, "v0", 200, [1.1])
    found = pick_tool.from_trainer_state(tmp_path)
    assert [step for step, _, _ in found] == [200]


def test_no_evaluations_yields_no_choice(tmp_path) -> None:
    assert pick_tool.from_trainer_state(tmp_path) == []
