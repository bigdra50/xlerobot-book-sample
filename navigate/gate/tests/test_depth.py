"""中央パッチの距離と高信頼の割合（純粋関数）。"""

from __future__ import annotations

import numpy as np
import pytest

from navgate.depth import CenterDepth, center_patch_depth


def test_center_patch_depth_reports_distance_even_when_no_pixel_is_high_confidence() -> None:
    # 実機で confidence が全画素 0 だった場面。距離は出し、高信頼の割合を 0 と報告する
    meters = np.full((32, 32), 2.0, dtype=np.float32)
    confidence = np.zeros((32, 32), dtype=np.uint8)
    actual = center_patch_depth(meters, confidence)
    assert actual == CenterDepth(median_m=2.0, high_ratio=0.0)


def test_center_patch_depth_counts_high_confidence_share_of_valid_pixels() -> None:
    meters = np.full((32, 32), 1.0, dtype=np.float32)
    confidence = np.zeros((32, 32), dtype=np.uint8)
    confidence[14:16, 14:18] = 2  # 中央 4×4 の半分
    actual = center_patch_depth(meters, confidence)
    assert actual is not None
    assert actual.median_m == pytest.approx(1.0)
    assert actual.high_ratio == pytest.approx(0.5)


def test_center_patch_depth_without_confidence_has_no_ratio() -> None:
    meters = np.full((16, 16), 0.4, dtype=np.float32)
    actual = center_patch_depth(meters, None)
    assert actual == CenterDepth(median_m=pytest.approx(0.4), high_ratio=None)


def test_center_patch_depth_all_invalid_returns_none() -> None:
    meters = np.zeros((24, 24), dtype=np.float32)
    confidence = np.full((24, 24), 2, dtype=np.uint8)
    assert center_patch_depth(meters, confidence) is None
