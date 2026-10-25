"""疑似 LaserScan を地図の占有格子に重ね、推定の位置が合っているかを確かめる。

rtabmap は、見え方の似た別の場所との一致を受け入れることがある。そのまま寄せ始めると、地図の上では
場所に着いたつもりで、実際には離れた場所に着く。

rtabmap の格子では、机の下や家具のまわりが中まで占有で埋まり、壁は線になっていない。点を壁の線に
合わせる判定は使えない。代わりに「推定が正しければ、点は空きのマスには落ちない」ことを使う。
推定がずれていれば、壁の手前の空きのマスに点が落ちる。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def free_hit_ratio(
    points: ArrayLike,
    cells: NDArray[np.integer],
    resolution: float,
    origin: tuple[float, float],
    pose: tuple[float, float, float],
) -> tuple[float, bool]:
    """base_link の点（N×2）を pose（x, y, yaw rad）に置いたとき、空き（値 0）のマスに落ちる割合と、
    台車の中心のマスが空きかどうか。格子の外と未知（-1）のマスは、空きに数えない。
    """
    p = np.asarray(points, dtype=np.float64)[:, :2]
    h, w = cells.shape
    ox, oy = origin
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    wx = x + c * p[:, 0] - s * p[:, 1]
    wy = y + s * p[:, 0] + c * p[:, 1]
    i = np.floor((wx - ox) / resolution).astype(int)
    j = np.floor((wy - oy) / resolution).astype(int)
    inside = (i >= 0) & (i < w) & (j >= 0) & (j < h)
    free = np.zeros(len(p), dtype=bool)
    free[inside] = cells[j[inside], i[inside]] == 0
    ci, cj = math.floor((x - ox) / resolution), math.floor((y - oy) / resolution)
    center_free = 0 <= ci < w and 0 <= cj < h and bool(cells[cj, ci] == 0)
    return float(np.mean(free)) if len(p) else 1.0, center_free
