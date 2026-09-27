from wafer_vlm.curate import arbitrate, geometry_verdict, structured_target, swift_rows


def record(failure_type: str, **features):
    base = {"radial_zone": "center", "clock_sector": 3, "resultant_length": 0.5}
    base.update(features)
    return {"sample_id": "s1", "failure_type": failure_type, "features": base}


def test_geometry_beats_the_wrong_teacher() -> None:
    rec = record("Edge_Ring", radial_zone="edge")
    wrong = {"radial_zone": "center", "shape": "环形"}
    right = {"radial_zone": "edge", "shape": "环形"}
    assert geometry_verdict(rec, right, "radial_zone")
    assert not geometry_verdict(rec, wrong, "radial_zone")
    assert arbitrate(rec, wrong, right, ["radial_zone"]) == "secondary"
    assert arbitrate(rec, right, wrong, ["radial_zone"]) == "primary"


def test_unresolvable_disagreement_stays_quarantined() -> None:
    # Edge_Ring permits several shapes, so shape alone cannot pick a winner.
    rec = record("Edge_Ring", radial_zone="edge")
    a = {"shape": "环形"}
    b = {"shape": "弧形"}
    assert geometry_verdict(rec, a, "shape") and geometry_verdict(rec, b, "shape")
    assert arbitrate(rec, a, b, ["shape"]) is None


def test_all_disputed_fields_must_point_at_one_teacher() -> None:
    rec = record("Scratch", radial_zone="middle", clock_sector=6)
    a = {"radial_zone": "middle", "clock_direction": 9}   # zone right, clock wrong
    b = {"radial_zone": "edge", "clock_direction": 6}     # zone wrong, clock right
    assert geometry_verdict(rec, a, "radial_zone") and not geometry_verdict(rec, a, "clock_direction")
    assert geometry_verdict(rec, b, "clock_direction") and not geometry_verdict(rec, b, "radial_zone")
    assert arbitrate(rec, a, b, ["radial_zone", "clock_direction"]) is None


def test_clock_tolerates_one_sector_and_rejects_all() -> None:
    rec = record("Scratch", radial_zone="middle", clock_sector=6)
    assert geometry_verdict(rec, {"clock_direction": 5}, "clock_direction")
    assert geometry_verdict(rec, {"clock_direction": 6}, "clock_direction")
    assert not geometry_verdict(rec, {"clock_direction": 9}, "clock_direction")
    assert not geometry_verdict(rec, {"clock_direction": "all"}, "clock_direction")


def test_clock_is_not_checked_when_geometry_is_ambiguous() -> None:
    # resultant_length < 0.25 means the centroid direction is meaningless.
    rec = record("Random", radial_zone="full", clock_sector=3, resultant_length=0.1)
    assert geometry_verdict(rec, {"clock_direction": "none"}, "clock_direction")
    assert not geometry_verdict(rec, {"clock_direction": 3}, "clock_direction")


def test_none_label_accepted_only_without_defect_zone() -> None:
    rec = record("none", radial_zone="full")
    assert geometry_verdict(rec, {"radial_zone": "none"}, "radial_zone")
    assert not geometry_verdict(rec, {"radial_zone": "center"}, "radial_zone")


def test_near_full_accepts_full_or_its_own_zone() -> None:
    rec = record("Near_full", radial_zone="center")
    assert geometry_verdict(rec, {"radial_zone": "full"}, "radial_zone")
    assert geometry_verdict(rec, {"radial_zone": "center"}, "radial_zone")
    assert not geometry_verdict(rec, {"radial_zone": "edge"}, "radial_zone")


TEACHER = {
    "defect_type": "Scratch", "shape": "线状", "radial_zone": "edge",
    "clock_direction": 11, "density": "稀疏", "continuity": "连续",
    "size_r": 0.4, "caption_zh": "中文描述", "caption_en": "english caption",
}


def test_geometry_gold_overrides_only_the_scored_geometry_fields() -> None:
    rec = record("Scratch", radial_zone="middle", clock_sector=6)
    rec["features"]["defect_extent_r"] = 0.812
    target = structured_target(rec, TEACHER, geometry_gold=True)
    # Scored against geometry gold, so taken from the trusted source.
    assert target["defect_type"] == "Scratch"
    assert target["radial_zone"] == "middle"
    assert target["clock_direction"] == 6
    assert target["size_r"] == 0.812
    # No geometric gold exists for these, so the teacher's wording survives.
    assert target["shape"] == "线状"
    assert target["density"] == "稀疏"
    assert target["continuity"] == "连续"
    assert target["caption_zh"] == "中文描述"
    assert target["caption_en"] == "english caption"


def test_teacher_structured_values_survive_when_flag_is_off() -> None:
    rec = record("Scratch", radial_zone="middle", clock_sector=6)
    target = structured_target(rec, TEACHER, geometry_gold=False)
    assert target["radial_zone"] == "edge"
    assert target["clock_direction"] == 11
    assert target["size_r"] == 0.4


def test_swift_rows_emits_three_tasks_per_sample() -> None:
    rec = record("Scratch", radial_zone="middle", clock_sector=6)
    rec["image_path"] = "/tmp/x.png"
    rows = swift_rows(rec, TEACHER, geometry_gold=True)
    assert [r["task"] for r in rows] == ["caption", "classification", "structured"]
    classification = rows[1]
    # The classification answer is the trusted label, never the teacher's guess.
    assert classification["messages"][-1]["content"] == "Scratch"
