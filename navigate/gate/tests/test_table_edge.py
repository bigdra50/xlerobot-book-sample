"""天板の手前の縁を深度の点から見つける（純粋関数）の仕様。

点は base_link（x 前、y 左、z 上、原点は床の上の回転の中心）で作る。
天板は下から見上げる形で、手前の縁の面と裏側の面だけを点にする（iPad のレンズは天板より低い）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navgate.table_edge import (
    Camera,
    Edge,
    approach_distance,
    docked_gap,
    find_edge,
    min_visible_distance,
    nearest_in_corridor,
    points_from_depth,
    wall_ahead,
)

TOP_Z = 0.73  # 天板の上面
THICK = 0.025  # 天板の厚み
STEP = 0.01


def _grid(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa, bb = np.meshgrid(a, b)
    return aa.ravel(), bb.ravel()


def _table(x0: float, slope: float, y_right: float, y_left: float, depth: float = 0.5) -> np.ndarray:
    """手前の縁が x = x0 + slope * y の天板。縁の面と裏側の面を点にする。"""
    y, z = _grid(np.arange(y_right, y_left, STEP), np.arange(TOP_Z - THICK, TOP_Z, STEP / 2))
    front = np.stack([x0 + slope * y, y, z], axis=-1)
    y2, d = _grid(np.arange(y_right, y_left, STEP), np.arange(0.0, depth, STEP))
    under = np.stack([x0 + slope * y2 + d, y2, np.full(y2.size, TOP_Z - THICK)], axis=-1)
    return np.concatenate([front, under])


def _wall(x: float, y_right: float = -0.8, y_left: float = 0.8) -> np.ndarray:
    y, z = _grid(np.arange(y_right, y_left, STEP), np.arange(0.0, 1.2, STEP))
    return np.stack([np.full(y.size, x), y, z], axis=-1)


def _column(x: float, y: float, width: float = 0.06) -> np.ndarray:
    yy, z = _grid(np.arange(y - width / 2, y + width / 2, STEP), np.arange(0.05, TOP_Z - THICK, STEP))
    return np.stack([np.full(yy.size, x), yy, z], axis=-1)


def test_square_table_in_front() -> None:
    pts = np.concatenate([_table(0.70, 0.0, -0.20, 0.20), _column(0.90, 0.0), _wall(1.40)])
    edge = find_edge(pts)
    assert edge is not None
    assert edge.offset == pytest.approx(0.70, abs=0.005)
    assert math.degrees(edge.yaw) == pytest.approx(0.0, abs=0.5)
    assert edge.distance == pytest.approx(0.70, abs=0.005)
    assert edge.y_right == pytest.approx(-0.20, abs=0.02)
    assert edge.y_left == pytest.approx(0.20, abs=0.02)
    assert edge.height == pytest.approx(TOP_Z - THICK, abs=0.02)
    assert edge.support_below < 0.2  # 縁の真下に点が無い（天板は張り出している）


@pytest.mark.parametrize("deg", [-12.0, 10.5])
def test_turned_table_gives_the_turn_to_square_up(deg: float) -> None:
    # 台車から見て縁が deg だけ傾いている。正対するには -deg ではなく、縁の法線へ回る
    slope = math.tan(math.radians(deg))
    pts = np.concatenate([_table(0.75, slope, -0.10, 0.30), _wall(1.45)])
    edge = find_edge(pts)
    assert edge is not None
    assert edge.slope == pytest.approx(slope, abs=0.01)
    assert math.degrees(edge.yaw) == pytest.approx(-deg, abs=0.5)
    assert edge.distance == pytest.approx(0.75 * math.cos(math.radians(deg)), abs=0.005)


def test_left_side_table_cut_by_the_view() -> None:
    # 部屋の隅の机。台車の左前にあり、左端は視野の外で切れている
    pts = np.concatenate([_table(0.70, 0.0, 0.28, 0.55), _wall(1.45, y_left=0.8)])
    edge = find_edge(pts)
    assert edge is not None
    assert edge.offset == pytest.approx(0.70, abs=0.005)
    assert edge.y_right == pytest.approx(0.28, abs=0.02)
    assert edge.y_left == pytest.approx(0.55, abs=0.02)


def test_stray_near_points_do_not_move_the_edge() -> None:
    rng = np.random.default_rng(0)
    stray = np.stack([rng.uniform(0.3, 0.5, 3), rng.uniform(-0.2, 0.2, 3), rng.uniform(0.6, 0.9, 3)], axis=-1)
    pts = np.concatenate([_table(0.70, 0.0, -0.20, 0.20), _wall(1.40), stray])
    edge = find_edge(pts)
    assert edge is not None
    assert edge.offset == pytest.approx(0.70, abs=0.005)


def test_wall_is_reported_as_supported_from_below() -> None:
    edge = find_edge(_wall(0.90))
    assert edge is not None
    assert edge.offset == pytest.approx(0.90, abs=0.005)
    assert edge.support_below > 0.8


def test_only_a_column_in_the_band_is_not_an_edge() -> None:
    # 縁が視野の上へ外れるほど近いと、帯に残るのは細い支柱だけになる
    assert find_edge(_column(0.35, 0.05)) is None


def test_nothing_in_the_band() -> None:
    floor_y, floor_x = _grid(np.arange(-0.5, 0.5, STEP), np.arange(1.0, 2.0, STEP))
    floor = np.stack([floor_x, floor_y, np.zeros(floor_x.size)], axis=-1)
    assert find_edge(floor) is None


# iPad の取り付け（光学 frame: z 前、x 右、y 下）を、yaw 0、pitch 0 で置いたもの
T_OPT = np.array(
    [
        [0.0, 0.0, 1.0, 0.128],
        [-1.0, 0.0, 0.0, 0.085],
        [0.0, -1.0, 0.0, 0.53],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
K = (213.0, 213.0, 127.5, 95.5)


def test_points_from_depth_drops_invalid_pixels_and_moves_to_base_link() -> None:
    depth = np.full((192, 256), 1.0)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    pts = points_from_depth(depth, K, T_OPT)
    assert pts.shape == (192 * 256 - 2, 3)
    # 画像の中心の画素は、レンズから真っ直ぐ前へ 1 m
    center = pts[np.argmin(np.hypot(pts[:, 1] - 0.085, pts[:, 2] - 0.53))]
    assert center == pytest.approx([1.128, 0.085, 0.53], abs=0.005)


def test_approach_and_docked_gap() -> None:
    edge = Edge(
        offset=0.70, slope=0.0, y_right=-0.2, y_left=0.2, height=0.7, bins=20, rms=0.002, support_below=0.0
    )
    # 前の縁（x = 0.175）と天板のあいだを 3 cm にするには、0.70 - 0.175 - 0.03 だけ進む
    assert approach_distance(edge, front_x=0.175, gap=0.03) == pytest.approx(0.495)
    # 0.45 m 下がってから測った縁が 0.655 m なら、止まっていたときの隙間は 0.655 - 0.45 - 0.175
    assert docked_gap(distance_after=0.655, backoff=0.45, front_x=0.175) == pytest.approx(0.03)


# --- 実際の深度に近い点: 画素ごとの光線を、床、壁、天板、支柱に当てて作る ---

W, H = 256, 192
K_IPAD = (213.17, 213.17, 126.77, 96.28)


def _mount(pitch_deg: float = 2.25, yaw_deg: float = 3.0) -> np.ndarray:
    """base_link から見た光学 frame。レンズは (0.128, 0.085, 0.53)、pitch は下向きが正。"""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    # link（x 前、y 左、z 上）: yaw のあと pitch
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    link_to_opt = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # 光学の軸を link で表す
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ link_to_opt
    T[:3, 3] = (0.128, 0.085, 0.53)
    return T


def _raycast(T: np.ndarray, table: dict | None, wall_x: float = 1.45, wall_deg: float = 0.0) -> np.ndarray:
    """画素ごとに最も近い面との交点（base_link の N×3）。天板は縁の中心 (x0, y0)、向き deg、幅、奥行きで置く。"""
    v, u = np.mgrid[0:H, 0:W]
    fx, fy, cx, cy = K_IPAD
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, dtype=float)], axis=-1).reshape(-1, 3)
    d = d @ T[:3, :3].T
    o = T[:3, 3]
    best = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d[:, 2] < 0, -o[2] / d[:, 2], np.inf)
        best = np.minimum(best, t_floor)
        wn = np.array([math.cos(math.radians(wall_deg)), math.sin(math.radians(wall_deg)), 0.0])
        t_wall = (wall_x - o @ wn) / (d @ wn)
        best = np.minimum(best, np.where(t_wall > 0, t_wall, np.inf))
        if table is not None:
            a = math.radians(table.get("deg", 0.0))
            n = np.array([math.cos(a), math.sin(a), 0.0])  # 縁の法線（天板の奥へ）
            e = np.array([-math.sin(a), math.cos(a), 0.0])  # 縁に沿って左へ
            c = np.array([table["x0"], table["y0"], 0.0])
            half = table.get("width", 0.42) / 2
            depth, top, thick = table.get("depth", 0.55), TOP_Z, THICK
            # 縁の面（鉛直）
            t_f = ((c - o) @ n) / (d @ n)
            pf = o + t_f[:, None] * d
            ok_f = (t_f > 0) & (np.abs((pf - c) @ e) <= half) & (pf[:, 2] >= top - thick) & (pf[:, 2] <= top)
            best = np.minimum(best, np.where(ok_f, t_f, np.inf))
            # 裏側（水平）
            t_u = (top - thick - o[2]) / d[:, 2]
            pu = o + t_u[:, None] * d
            w = (pu - c) @ n
            ok_u = (t_u > 0) & (w >= 0) & (w <= depth) & (np.abs((pu - c) @ e) <= half)
            best = np.minimum(best, np.where(ok_u, t_u, np.inf))
            # 支柱（縁から奥へ 0.2 m、幅 6 cm の前面）
            cc = c + 0.2 * n
            t_c = ((cc - o) @ n) / (d @ n)
            pc = o + t_c[:, None] * d
            ok_c = (t_c > 0) & (np.abs((pc - cc) @ e) <= 0.03) & (pc[:, 2] <= top - thick)
            best = np.minimum(best, np.where(ok_c, t_c, np.inf))
    ok = np.isfinite(best)
    return o + best[ok, None] * d[ok]


def _camera(T: np.ndarray) -> Camera:
    return Camera(K=K_IPAD, width=W, height=H, T_base_opt=T)


def test_raycast_square_table() -> None:
    T = _mount()
    pts = _raycast(T, {"x0": 0.80, "y0": 0.0})
    edge = find_edge(pts, camera=_camera(T))
    assert edge is not None
    assert edge.distance == pytest.approx(0.80, abs=0.01)
    assert math.degrees(edge.yaw) == pytest.approx(0.0, abs=1.0)
    assert not edge.clipped_left and not edge.clipped_right


def test_raycast_corner_table_cut_by_the_view_keeps_the_true_line() -> None:
    # 左前の隅の机。左の部分は視野の外で、その先は天板の裏側が視野の端に沿って奥へ続いて見える
    T = _mount()
    pts = _raycast(T, {"x0": 0.72, "y0": 0.50, "deg": 10.5, "width": 0.42})
    edge = find_edge(pts, camera=_camera(T))
    assert edge is not None
    # 縁の法線が左へ 10.5° 向いているので、正対するには左へ 10.5° 回る
    assert math.degrees(edge.yaw) == pytest.approx(10.5, abs=1.5)
    assert edge.clipped_left
    # 縁の線の上の、机の中心（y = 0.50）の点
    assert edge.offset + edge.slope * 0.50 == pytest.approx(0.72, abs=0.01)


def test_raycast_edge_above_the_view_is_not_taken_for_the_edge() -> None:
    # 近すぎて縁が視野の上へ外れると、見えるのは裏側の奥だけになる。
    # 裏側の区切りは視野の端で切れているので捨て、残るのは机の横に見える奥の壁になる
    T = _mount()
    pts = _raycast(T, {"x0": 0.50, "y0": 0.0})
    edge = find_edge(pts, camera=_camera(T))
    assert edge is not None
    assert edge.distance == pytest.approx(1.45, abs=0.01)
    assert edge.support_below > 0.8


def test_min_visible_distance_for_the_edge_height() -> None:
    cam = Camera(K=K, width=256, height=192, T_base_opt=T_OPT)
    # 上から 4 画素の行は、水平から atan((95.5 - 4) / 213) 上を見る。縁はレンズより 0.20 m 高い
    expected = 0.20 / ((95.5 - 4.0) / 213.0) + 0.03
    assert min_visible_distance(cam, 0.73) == pytest.approx(expected, abs=0.001)
    # レンズより低い縁は、上へ外れない
    assert min_visible_distance(cam, 0.40) == pytest.approx(0.03)


def test_wall_ahead_under_the_table() -> None:
    # 天板の縁が視野の上へ外れるほど近くても、縁の下の奥の壁は見え続ける
    T = _mount()
    for x0 in (0.80, 0.45):
        pts = _raycast(T, {"x0": x0, "y0": 0.0}, wall_x=1.45)
        assert wall_ahead(pts, lens_y=0.085, beyond=x0 + 0.1) == pytest.approx(1.45, abs=0.005)


def test_wall_ahead_needs_enough_points() -> None:
    T = _mount()
    pts = _raycast(T, {"x0": 0.80, "y0": 0.0}, wall_x=1.45)
    assert wall_ahead(pts, lens_y=0.085, beyond=1.5) is None


def test_narrow_post_in_front_of_the_table_does_not_hide_the_edge() -> None:
    # 天板を支える柱が縁より手前に立っていて、帯の高さにも掛かっている
    post = _column(0.60, 0.40, width=0.08)
    pts = np.concatenate([_table(0.70, 0.0, -0.20, 0.33), post, _wall(1.40)])
    edge = find_edge(pts)
    assert edge is not None
    assert edge.offset == pytest.approx(0.70, abs=0.005)
    assert edge.y_left == pytest.approx(0.32, abs=0.02)


def test_nearest_in_corridor() -> None:
    table = _table(0.70, 0.0, -0.20, 0.40)
    floor_y, floor_x = _grid(np.arange(-0.5, 0.5, STEP), np.arange(0.3, 1.5, STEP))
    floor = np.stack([floor_x, floor_y, np.zeros(floor_x.size)], axis=-1)
    # 柱が台車の通り道の外（y = 0.40）なら、一番手前は縁
    outside = np.concatenate([table, floor, _column(0.60, 0.40)])
    assert nearest_in_corridor(outside, half_width=0.25) == pytest.approx(0.70, abs=0.005)
    # 通り道の中（y = 0.20）なら、柱
    inside = np.concatenate([table, floor, _column(0.60, 0.20)])
    assert nearest_in_corridor(inside, half_width=0.25) == pytest.approx(0.60, abs=0.005)
    assert nearest_in_corridor(floor, half_width=0.25) is None
