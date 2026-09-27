"""Loader for the Python 2-era WM811K ``LSWMD.pkl`` artefact.

``LSWMD.pkl`` was serialised with Python 2 and pandas 0.x. Its pickle stream
refers to ``pandas.indexes.*``, a package pandas removed in 1.0, so
``pandas.read_pickle`` fails on a modern pandas with ``ModuleNotFoundError``.
The default ASCII codec also cannot decode the byte-string metadata.

Both problems are worked around here rather than by pinning an ancient pandas,
which would not build on Python 3.12. ``load_lswmd_frame`` is the only supported
entry point; it installs the shims, loads with the ``latin1`` codec and returns
the frame with its original column names intact.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import pandas as pd

#: Legacy module path (below ``pandas.indexes``) -> the classes such a pickle can
#: name. ``numeric`` lost its concrete index types in pandas 2.0, so those map
#: onto the plain ``Index``.
_LEGACY_INDEX_MODULES: dict[str, dict[str, type]] = {
    "base": {"Index": pd.Index},
    "range": {"RangeIndex": pd.RangeIndex},
    "numeric": {
        "Int64Index": pd.Index,
        "Float64Index": pd.Index,
        "UInt64Index": pd.Index,
        "NumericIndex": pd.Index,
    },
    "category": {"CategoricalIndex": pd.CategoricalIndex},
    "datetimes": {"DatetimeIndex": pd.DatetimeIndex},
    "timedeltas": {"TimedeltaIndex": pd.TimedeltaIndex},
    "period": {"PeriodIndex": pd.PeriodIndex},
    "interval": {"IntervalIndex": pd.IntervalIndex},
    "multi": {"MultiIndex": pd.MultiIndex},
}

#: Attributes old pandas stored on an index that no longer exist.
_STALE_INDEX_ATTRS = ("_typ", "_reset_cache", "_cache", "copy")


def _new_Index(cls, d):  # noqa: N802 - name is part of the pickle stream
    """Rebuild an index from the attribute dict an old pickle stored.

    Old pandas serialised indexes as ``(_new_Index, (cls, attributes))``. The
    attributes carry the same keywords the modern constructors accept, so the
    direct call is tried first and a plain ``Index`` is built as a fallback.
    """
    if not isinstance(d, dict):
        return pd.Index(d)
    attrs = {k: v for k, v in d.items() if k not in _STALE_INDEX_ATTRS}
    try:
        return cls(**attrs)
    except Exception:
        try:
            return pd.Index(attrs.get("data", []), name=attrs.get("name"))
        except Exception:
            return pd.Index([])


def install_legacy_pandas_shims() -> None:
    """Register the ``pandas.indexes.*`` aliases an old pickle resolves against.

    Idempotent, and a no-op for any path pandas still provides itself.
    """
    package = types.ModuleType("pandas.indexes")
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault("pandas.indexes", package)

    for submodule, classes in _LEGACY_INDEX_MODULES.items():
        name = f"pandas.indexes.{submodule}"
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        module._new_Index = _new_Index  # type: ignore[attr-defined]
        for class_name, cls in classes.items():
            setattr(module, class_name, cls)
        sys.modules.setdefault(name, module)
        if getattr(package, submodule, None) is None:
            setattr(package, submodule, module)


def load_lswmd_frame(path: str | Path) -> pd.DataFrame:
    """Load ``LSWMD.pkl`` into a DataFrame on a modern pandas.

    Raises ``FileNotFoundError`` if the artefact is absent and ``TypeError`` if
    the pickle does not hold a DataFrame.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LSWMD pickle not found: {source}")

    install_legacy_pandas_shims()
    with source.open("rb") as handle:
        frame = pickle.load(handle, encoding="latin1")  # noqa: S301 - trusted local artefact

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {source}, got {type(frame).__name__}")
    return frame
