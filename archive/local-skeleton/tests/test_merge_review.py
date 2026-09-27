import importlib.util
from pathlib import Path


def _find_tool(name: str) -> Path:
    """Locate a tool script whether tools/ sits beside the project or above it.

    The deployment keeps the project nested under projects/ with a shared
    tools/ directory at the repository root, so a fixed relative path only
    works in one of the two layouts.
    """
    for base in Path(__file__).resolve().parents:
        candidate = base / "tools" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


_spec = importlib.util.spec_from_file_location("merge_review", _find_tool("merge_review.py"))
merge_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_review)


def test_perfect_agreement_scores_one() -> None:
    assert merge_review.cohen_kappa([("Center", "Center"), ("Loc", "Loc")]) == 1.0


def test_agreement_that_is_purely_chance_is_undefined() -> None:
    # Both raters used a single identical label, so chance agreement is 1 and
    # kappa has no denominator; reporting None beats reporting a fake number.
    assert merge_review.cohen_kappa([("Center", "Center")]) is None


def test_systematic_disagreement_scores_negative() -> None:
    kappa = merge_review.cohen_kappa([("a", "b"), ("b", "a"), ("a", "b"), ("b", "a")])
    assert kappa is not None and kappa < 0


def test_empty_sheet_scores_none() -> None:
    assert merge_review.cohen_kappa([]) is None


def test_normalise_treats_blank_and_whitespace_as_unlabelled() -> None:
    assert merge_review.normalise(None) == ""
    assert merge_review.normalise("   ") == ""
    assert merge_review.normalise(" Loc ") == "Loc"
