"""壁に貼ったマーカーと、その壁の面から、登録した位置へ台車を戻す 1 手を求める。

位置はマーカーの中心から取る。35 mm のマーカーでも、中心は 30 枚で 0.2 mm しかばらつかなかった。
向きはマーカーから取らない。小さい正方形の PnP は面の向きが ±12° ばらつき、裏返った解も出たため。
代わりに、マーカーのまわりの壁を深度の点で直線に当てはめ、壁に対する台車の向きを求める。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class MarkerRef:
    """登録した位置で見た、base_link の座標のマーカーの中心 [m] と、壁に対する台車の向き [rad]。"""

    marker_id: int
    marker_xy: tuple[float, float]
    wall_yaw: float


def wall_slope(
    points: ArrayLike,
    *,
    marker_xy: tuple[float, float],
    marker_z: float,
    half_width: float = 0.30,
    half_height: float = 0.15,
    depth_band: float = 0.08,
    min_points: int = 30,
) -> float | None:
    """マーカーのまわりの壁の点（base_link、N×3）を x = a + b y に当てはめ、壁に対する台車の向き atan(b) を返す。

    壁の法線の向きを 0 とし、台車が左回りに回っていれば正。点が足りなければ None。
    手前の物を拾わないように、マーカーと同じ奥行きの帯（±depth_band）の点だけを使い、外れた点を除いて当てはめ直す。
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or len(p) < min_points:
        return None
    mx, my = marker_xy
    sel = (
        (np.abs(p[:, 1] - my) <= half_width)
        & (np.abs(p[:, 2] - marker_z) <= half_height)
        & (np.abs(p[:, 0] - mx) <= depth_band)
    )
    q = p[sel]
    for _ in range(2):
        if len(q) < min_points:
            return None
        b, a = np.polyfit(q[:, 1], q[:, 0], 1)
        resid = np.abs(q[:, 0] - (a + b * q[:, 1]))
        keep = resid <= max(0.01, 2.5 * float(np.median(resid)))
        if keep.all():
            break
        q = q[keep]
    return math.atan(b)


def align_step(marker_xy: tuple[float, float], wall_yaw: float, ref: MarkerRef) -> tuple[float, float, float]:
    """今見ているマーカーの中心と壁の向きから、(回る角 [rad], 前 [m], 左 [m]) を返す。

    先にその場で回り、回ったあとの向きで前と左へ動くと、登録した位置に戻る。
    """
    dth = ref.wall_yaw - wall_yaw
    c, s = math.cos(dth), math.sin(dth)
    mx, my = marker_xy
    rx, ry = c * mx + s * my, -s * mx + c * my  # 回ったあとの座標で見たマーカー
    return dth, rx - ref.marker_xy[0], ry - ref.marker_xy[1]
