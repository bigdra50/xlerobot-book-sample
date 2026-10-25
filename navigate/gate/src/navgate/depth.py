"""深度画像の中央パッチの距離と、高信頼の画素の割合を出す（純粋関数）。

LiDAR が生きている確認に使う。無効な画素は 0 か NaN。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# ARKit の confidence は 0 / 1 / 2 で 2 が最高
CONFIDENCE_HIGH: Final = 2
CENTER_FRACTION: Final = 1 / 8


def _center_slices(height: int, width: int, fraction: float) -> tuple[slice, slice]:
    patch_h, patch_w = max(1, int(height * fraction)), max(1, int(width * fraction))
    y0, x0 = (height - patch_h) // 2, (width - patch_w) // 2
    return slice(y0, y0 + patch_h), slice(x0, x0 + patch_w)


@dataclass(frozen=True)
class CenterDepth:
    """中央パッチの距離と、そのうち高信頼の画素の割合。

    Attributes:
        median_m: 有限かつ正の画素の中央値（m）。confidence は問わない。
        high_ratio: 有効な画素のうち confidence 2 の割合。confidence が無ければ None。
    """

    median_m: float
    high_ratio: float | None


def center_patch_depth(
    meters: NDArray[Any],
    confidence: NDArray[Any] | None,
    fraction: float = CENTER_FRACTION,
) -> CenterDepth | None:
    """LiDAR が生きているかを見るための距離。高信頼の画素が無くても距離は返す。

    実機では、場面によって confidence が全画素 0 で届く。高信頼だけに絞ると「深度が来ていない」と区別できない。
    """
    height, width = meters.shape[:2]
    ys, xs = _center_slices(height, width, fraction)
    patch = np.asarray(meters[ys, xs], dtype=np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    if not valid.any():
        return None
    high_ratio = None
    if confidence is not None:
        high = np.asarray(confidence[ys, xs]) == CONFIDENCE_HIGH
        high_ratio = float((high & valid).sum() / valid.sum())
    return CenterDepth(median_m=float(np.median(patch[valid])), high_ratio=high_ratio)
