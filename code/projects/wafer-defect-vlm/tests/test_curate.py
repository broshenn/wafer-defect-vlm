from wafer_vlm.curate import core_disagreement, validate


def record() -> dict:
    return {
        "sample_id": "wafer_1_1", "failure_type": "Center",
        "features": {"radial_zone": "center", "clock_sector": 1, "resultant_length": 0.1},
    }


def annotation(label: str = "Center") -> dict:
    return {"result": {
        "defect_type": label, "shape": "团簇状", "radial_zone": "center",
        "clock_direction": "none", "density": "致密", "continuity": "连续",
        "size_r": 0.3, "caption_zh": "一个致密的中心团簇。",
        "caption_en": "A dense central cluster.", "uncertain": False,
    }}


def test_valid_teacher_passes() -> None:
    result = validate(record(), annotation())
    assert result["passed"]
    assert result["confidence"] == 100


def test_type_mismatch_is_hard_failure() -> None:
    result = validate(record(), annotation("Donut"))
    assert not result["passed"]
    assert any(issue.startswith("type_mismatch") for issue in result["issues"])


def test_core_disagreement_detects_shape() -> None:
    first = annotation()["result"]
    second = {**first, "shape": "满圆"}
    assert "shape" in core_disagreement(first, second)
