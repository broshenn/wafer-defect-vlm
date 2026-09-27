"""Tests for the legacy ``LSWMD.pkl`` loader.

The real artefact is a 2 GB Python 2 pickle, so these tests synthesise a pickle
that names the same removed module paths instead.
"""

import io
import pickle
import sys

import pandas as pd
import pytest

from wafer_vlm.lswmd import _new_Index, install_legacy_pandas_shims, load_lswmd_frame


@pytest.fixture
def legacy_pickler_path():
    """Make ``_new_Index`` picklable under the path pandas 0.x wrote.

    ``pickle`` refuses to dump a function unless the object it finds at the
    recorded module path is the same object, so the shim has to be installed
    first and ``_new_Index`` has to advertise the legacy path.
    """
    install_legacy_pandas_shims()
    original = (_new_Index.__module__, _new_Index.__qualname__)
    _new_Index.__module__ = "pandas.indexes.base"
    _new_Index.__qualname__ = "_new_Index"
    try:
        yield
    finally:
        _new_Index.__module__, _new_Index.__qualname__ = original


class _LegacyIndexPickler(pickle.Pickler):
    """Serialises indexes the way pandas 0.x did, via ``pandas.indexes.base``."""

    def reducer_override(self, obj):
        if isinstance(obj, pd.RangeIndex):
            attrs = {"start": obj.start, "stop": obj.stop, "step": obj.step, "name": obj.name}
            return (_new_Index, (pd.RangeIndex, attrs))
        if type(obj) is pd.Index:
            return (_new_Index, (pd.Index, {"data": obj.tolist(), "name": obj.name}))
        return NotImplemented


def test_shims_expose_modern_equivalents() -> None:
    install_legacy_pandas_shims()

    assert sys.modules["pandas.indexes.base"].Index is pd.Index
    assert sys.modules["pandas.indexes.numeric"].Int64Index is pd.Index
    assert sys.modules["pandas.indexes.base"]._new_Index is _new_Index


def test_shims_are_idempotent() -> None:
    install_legacy_pandas_shims()
    first = sys.modules["pandas.indexes.base"]
    install_legacy_pandas_shims()

    assert sys.modules["pandas.indexes.base"] is first


def test_new_index_rebuilds_a_range_index() -> None:
    index = _new_Index(pd.RangeIndex, {"start": 0, "stop": 4, "step": 1, "name": None})

    assert list(index) == [0, 1, 2, 3]


def test_new_index_ignores_attributes_modern_pandas_dropped() -> None:
    index = _new_Index(pd.Index, {"data": ["a", "b"], "name": "lot", "_typ": "index", "_cache": {}})

    assert list(index) == ["a", "b"]
    assert index.name == "lot"


def test_legacy_pickle_round_trips_through_the_shims(legacy_pickler_path) -> None:
    frame = pd.DataFrame({"failureType": ["Center", "Donut", "none"]}, index=pd.Index(["L1", "L2", "L3"], name="lot"))
    buffer = io.BytesIO()
    _LegacyIndexPickler(buffer).dump(frame)

    restored = pickle.loads(buffer.getvalue(), encoding="latin1")

    assert list(restored.index) == ["L1", "L2", "L3"]
    assert restored.index.name == "lot"
    assert list(restored["failureType"]) == ["Center", "Donut", "none"]


def test_missing_artifact_names_the_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="LSWMD pickle not found"):
        load_lswmd_frame(tmp_path / "LSWMD.pkl")
