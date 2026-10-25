"""地図から巡回の点を決める（純粋関数）。

巡回の点には、台車が実際にその場で回れた場所を使う。
地図の空きだけで選ぶと、すりガラスのように深度が通り抜ける物の向こうへ点を置くことがあるため。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np


def turn_spots(
    poses: Sequence[tuple[float, float, float]], radius: float = 0.05, min_turn_deg: float = 300.0
) -> list[tuple[float, float]]:
    """通った道（x、y、yaw の並び）のうち、その場で min_turn_deg 以上回った場所を、通った順に返す。

    続けて並ぶ姿勢のうち、最初の姿勢から radius [m] 以内にあるものを 1 つの場所とみなし、
    その場所での yaw の広がり（±π の折り返しをほどいた値）を回った角とする。位置は場所の平均。
    """
    spots = []
    group: list[tuple[float, float, float]] = []

    def close(g: list[tuple[float, float, float]]) -> None:
        if len(g) < 2:
            return
        yaws = np.unwrap([p[2] for p in g])
        if math.degrees(float(yaws.max() - yaws.min())) >= min_turn_deg:
            spots.append((float(np.mean([p[0] for p in g])), float(np.mean([p[1] for p in g]))))

    for pose in poses:
        if group and math.hypot(pose[0] - group[0][0], pose[1] - group[0][1]) > radius:
            close(group)
            group = []
        group.append(pose)
    close(group)
    return spots


def loop_route(
    start: tuple[float, float], points: Sequence[tuple[float, float]]
) -> list[tuple[float, float, float]]:
    """台車（start）に近い点から、いちばん近い点を順にたどる巡回の順を返す。

    各点の yaw は、巡回で次に行く点の方向（最後の点は最初の点の方向）。
    """
    left = list(points)
    order = []
    here = start
    while left:
        nearest = min(left, key=lambda p: math.hypot(p[0] - here[0], p[1] - here[1]))
        order.append(nearest)
        left.remove(nearest)
        here = nearest
    route = []
    for k, (x, y) in enumerate(order):
        nx, ny = order[(k + 1) % len(order)]
        route.append((x, y, math.atan2(ny - y, nx - x) if (nx, ny) != (x, y) else 0.0))
    return route


def clearance_at(
    grid: np.ndarray, resolution: float, origin: tuple[float, float], x: float, y: float
) -> float:
    """(x, y) のマスの中心から、いちばん近い障害物（50 以上）か未知（負）のマスの中心までの距離 [m]。

    grid は nav_msgs/OccupancyGrid と同じ並び（行が y、列が x）。(x, y) が障害物、未知、地図の外なら 0。
    """
    i = math.floor((y - origin[1]) / resolution)
    j = math.floor((x - origin[0]) / resolution)
    if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
        return 0.0
    blocked = np.argwhere((grid >= 50) | (grid < 0))
    if len(blocked) == 0:
        return math.inf
    return float(np.hypot(blocked[:, 0] - i, blocked[:, 1] - j).min() * resolution)


def parse_points(value: object) -> list[tuple[float, float]]:
    """点の並びを [(x, y), ...] にする。

    JSON の文字列（[[x, y], ...] か [x1, y1, x2, y2, ...]）と、ROS の数の配列（入れ子にできないので平らな並び）を受ける。
    空の文字列、空の配列、None は点なし。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        value = json.loads(value)
    flat = np.asarray(value, dtype=float).ravel()
    if flat.size % 2:
        raise ValueError(f"点は x, y の組で渡す（値が {flat.size} 個ある）")
    return [(float(x), float(y)) for x, y in flat.reshape(-1, 2)]
