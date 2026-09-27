"""Deterministic GRPO rewards for the wafer structured-output task.

Every reward is computed from the trusted class label and the deterministic
geometry features that ship with the sample. Nothing here calls a model, an API
or a judge: an LLM judge used as the reward would let the policy drift toward
whatever the judge happens to like, and the gold label and the circle-fit
geometry are the only two sources this project treats as trustworthy.

Register with ``--external_plugins <this file>`` and select with
``--reward_funcs wafer_class wafer_format wafer_radial wafer_clock``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from swift.rewards import ORM, orms

SCORED_FIELDS = ("defect_type", "shape", "radial_zone", "clock_direction", "density", "continuity", "size_r")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the last balanced ``{...}`` block of ``text`` that parses as an object.

    A fenced block is preferred when present, because a model that wraps its
    answer in ```json ... ``` still gave a usable answer and scoring it as a
    format failure would train away a correct reply.
    """
    if not text:
        return None
    blocks = _FENCE.findall(text)
    blocks += _balanced_objects(text)
    for blob in reversed(blocks):
        try:
            value = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _balanced_objects(text: str) -> list[str]:
    found: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start:end + 1])
                    break
    return found


def parse_gold(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def circular_distance(a: Any, b: Any) -> int | None:
    """Clock sectors apart on a 1-12 dial."""
    try:
        diff = abs(int(a) - int(b)) % 12
    except (TypeError, ValueError):
        return None
    return min(diff, 12 - diff)


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


class _WaferReward(ORM):
    """Shared plumbing: pair each completion with its parsed gold."""

    def pairs(self, completions, ground_truth):
        for completion, gold_text in zip(completions, ground_truth):
            yield completion, parse_gold(gold_text)


class WaferClass(_WaferReward):
    """1.0 when the reported defect class matches the trusted label."""

    def __call__(self, completions, ground_truth, **kwargs):
        rewards = []
        for completion, gold in self.pairs(completions, ground_truth):
            answer = extract_json(completion)
            if gold is None or answer is None:
                rewards.append(0.0)
                continue
            rewards.append(
                1.0 if _as_text(answer.get("defect_type")) == _as_text(gold.get("defect_type")) else 0.0)
        return rewards


class WaferFormat(_WaferReward):
    """1.0 when the reply is a single JSON object carrying every scored field."""

    def __call__(self, completions, ground_truth, **kwargs):
        rewards = []
        for completion, gold in self.pairs(completions, ground_truth):
            answer = extract_json(completion)
            if gold is None or answer is None:
                rewards.append(0.0)
                continue
            present = sum(1 for field in SCORED_FIELDS if field in answer)
            rewards.append(present / len(SCORED_FIELDS))
        return rewards


class WaferRadial(_WaferReward):
    """1.0 when ``radial_zone`` matches the deterministic geometry.

    Returns a constant 1.0 when the sample has no geometric gold, so an
    unscoreable field contributes no advantage rather than a free point that
    would reward guessing.
    """

    def __call__(self, completions, ground_truth, **kwargs):
        rewards = []
        for completion, gold in self.pairs(completions, ground_truth):
            if gold is None:
                rewards.append(0.0)
                continue
            expected = gold.get("radial_zone")
            if not expected:
                rewards.append(1.0)
                continue
            answer = extract_json(completion)
            rewards.append(1.0 if answer is not None and _as_text(answer.get("radial_zone")) == _as_text(expected)
                          else 0.0)
        return rewards


class WaferClock(_WaferReward):
    """1.0 when the reported clock sector is within one of the geometry gold."""

    def __call__(self, completions, ground_truth, **kwargs):
        rewards = []
        for completion, gold in self.pairs(completions, ground_truth):
            if gold is None:
                rewards.append(0.0)
                continue
            expected = gold.get("clock_direction")
            if expected in (None, "", "all", "none"):
                rewards.append(1.0)
                continue
            answer = extract_json(completion)
            if answer is None:
                rewards.append(0.0)
                continue
            distance = circular_distance(answer.get("clock_direction"), expected)
            rewards.append(1.0 if distance is not None and distance <= 1 else 0.0)
        return rewards


orms['wafer_class'] = WaferClass
orms['wafer_format'] = WaferFormat
orms['wafer_radial'] = WaferRadial
orms['wafer_clock'] = WaferClock
