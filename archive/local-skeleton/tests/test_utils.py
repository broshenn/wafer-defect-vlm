import numpy as np
import pandas as pd

from wafer_vlm.utils import (
    calculate_static_features,
    normalize_label,
    normalize_label_series,
    split_for_lot,
)


def disk(size: int = 31) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[(xx - center) ** 2 + (yy - center) ** 2 <= (center - 1) ** 2] = 1
    return arr


def test_clock_is_clockwise_from_twelve() -> None:
    wafer = disk()
    wafer[4:6, 14:17] = 2
    features = calculate_static_features(wafer)
    assert features["clock_sector"] in {12, 1}
    assert features["centroid_xy_r"][1] > 0


def test_isolated_noise_is_explicitly_reported() -> None:
    wafer = disk()
    wafer[15, 15] = 2
    features = calculate_static_features(wafer)
    assert features["status"] == "filtered_empty"
    assert features["raw_defect_count"] == 1
    assert features["defect_count"] == 0
    assert features["filtered_points_removed"] == 1


def test_label_and_lot_split_are_stable() -> None:
    assert normalize_label(np.array([["Edge-Ring"]], dtype=object)) == "Edge_Ring"
    assert split_for_lot("lot123", 3407) == split_for_lot("lot123", 3407)


def test_label_series_keeps_unlabelled_rows_as_none() -> None:
    column = pd.Series([
        np.array([["Edge-Ring"]], dtype=object),
        np.array([], dtype=object),
        np.array([["none"]], dtype=object),
    ])

    labels = normalize_label_series(column)

    assert labels == ["Edge_Ring", None, "none"]
    assert [x for x in labels if x is not None] == ["Edge_Ring", "none"]
