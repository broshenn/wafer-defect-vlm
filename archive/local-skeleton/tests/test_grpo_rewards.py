import importlib.util
import json
from pathlib import Path

import pytest


def _find_tool(name: str) -> Path:
    for base in Path(__file__).resolve().parents:
        candidate = base / "tools" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


pytest.importorskip("swift", reason="reward plugins import ms-swift's ORM base class")

_spec = importlib.util.spec_from_file_location("wafer_grpo_plugin", _find_tool("wafer_grpo_plugin.py"))
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)

GOLD = json.dumps({"defect_type": "Loc", "radial_zone": "center", "clock_direction": 3})


def test_extract_json_prefers_a_fenced_block() -> None:
    text = 'reasoning here\n```json\n{"defect_type": "Loc"}\n```'
    assert plugin.extract_json(text) == {"defect_type": "Loc"}


def test_extract_json_ignores_braces_that_are_not_json() -> None:
    # A model that shows its working will mention braces; only the answer counts.
    text = 'The set {a, b} is irrelevant. Answer: {"defect_type": "Donut"}'
    assert plugin.extract_json(text) == {"defect_type": "Donut"}


def test_extract_json_takes_the_last_object_when_several_parse() -> None:
    text = '{"defect_type": "Loc"} then corrected: {"defect_type": "Center"}'
    assert plugin.extract_json(text) == {"defect_type": "Center"}


def test_extract_json_returns_none_for_prose() -> None:
    assert plugin.extract_json("I cannot tell from this image.") is None
    assert plugin.extract_json("") is None


def test_circular_distance_wraps_across_twelve() -> None:
    assert plugin.circular_distance(12, 1) == 1
    assert plugin.circular_distance(1, 12) == 1
    assert plugin.circular_distance(1, 7) == 6
    assert plugin.circular_distance(None, 3) is None


def test_class_reward_scores_the_trusted_label() -> None:
    completions = ['{"defect_type": "Loc"}', '{"defect_type": "Donut"}', "no json at all"]
    assert plugin.WaferClass()(completions, [GOLD] * 3, ) == [1.0, 0.0, 0.0]


def test_format_reward_is_fractional_over_the_scored_fields() -> None:
    completions = ['{"defect_type": "Loc", "shape": "x", "radial_zone": "center", "clock_direction": 3,'
                   ' "density": "d", "continuity": "c", "size_r": 0.3}', "nonsense"]
    rewards = plugin.WaferFormat()(completions, [GOLD] * 2)
    assert rewards == [1.0, 0.0]


def test_radial_reward_is_silent_when_geometry_has_no_zone() -> None:
    # An unscoreable field must not hand out a free point for guessing.
    gold = json.dumps({"defect_type": "Loc", "radial_zone": None, "clock_direction": None})
    assert plugin.WaferRadial()(["{}"], [gold]) == [1.0]
    assert plugin.WaferClock()(["{}"], [gold]) == [1.0]


def test_radial_reward_checks_against_geometry() -> None:
    completions = ['{"radial_zone": "center"}', '{"radial_zone": "edge"}']
    assert plugin.WaferRadial()(completions, [GOLD] * 2) == [1.0, 0.0]


def test_clock_reward_allows_one_sector_of_slack() -> None:
    completions = ['{"clock_direction": 3}', '{"clock_direction": 4}', '{"clock_direction": 6}']
    assert plugin.WaferClock()(completions, [GOLD] * 3) == [1.0, 1.0, 0.0]


def test_rewards_are_registered_under_their_cli_names() -> None:
    from swift.rewards import orms
    for name in ("wafer_class", "wafer_format", "wafer_radial", "wafer_clock"):
        assert name in orms
