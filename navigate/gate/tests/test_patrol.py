"""地図から巡回の点を決める（純粋関数）の仕様。

- 回った場所: 地図の通った道のうち、同じ場所でその場の旋回をした区間
- 巡回の順: 台車に近い点から近いものをたどり、各点で次の点の方を向く
- 余裕: 点から、いちばん近い障害物か未知のマスまでの距離
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navgate.patrol import clearance_at, loop_route, parse_points, turn_spots


def _spin(x: float, y: float, degrees: float, n: int = 20) -> list[tuple[float, float, float]]:
    """(x, y) で 0 から degrees まで回る。yaw は ±π に折り返す。"""
    out = []
    for k in range(n):
        yaw = math.radians(degrees * k / (n - 1))
        out.append((x + 0.004 * (k % 2), y, math.atan2(math.sin(yaw), math.cos(yaw))))
    return out


def _drive(x0: float, y0: float, x1: float, y1: float, n: int = 10) -> list[tuple[float, float, float]]:
    yaw = math.atan2(y1 - y0, x1 - x0)
    return [(x0 + (x1 - x0) * k / (n - 1), y0 + (y1 - y0) * k / (n - 1), yaw) for k in range(n)]


def test_a_full_turn_in_place_is_a_turn_spot() -> None:
    spots = turn_spots(_spin(1.0, 2.0, 360))
    assert len(spots) == 1
    assert spots[0] == pytest.approx((1.002, 2.0), abs=1e-3)


def test_driving_straight_has_no_turn_spot() -> None:
    assert turn_spots(_drive(0.0, 0.0, 1.0, 0.0)) == []


def test_a_half_turn_is_not_enough() -> None:
    assert turn_spots(_spin(0.0, 0.0, 180)) == []


def test_spots_come_in_travel_order() -> None:
    poses = _spin(0.0, 0.0, 360) + _drive(0.0, 0.0, 0.0, -1.0) + _spin(0.0, -1.0, 400)
    spots = turn_spots(poses)
    assert [round(y, 2) for _, y in spots] == [0.0, -1.0]


def test_crossing_pi_counts_the_short_way() -> None:
    # -170° から +170° へは 20° しか回っていない
    poses = [(0.0, 0.0, math.radians(a)) for a in (-170, -175, 180, 175, 170)] * 4
    assert turn_spots(poses) == []


def test_loop_starts_near_the_robot_and_faces_the_next_point() -> None:
    points = [(0.0, -1.0), (0.0, 0.3), (-1.0, 0.6)]
    route = loop_route((0.0, -0.3), points)
    assert [(x, y) for x, y, _ in route] == [(0.0, 0.3), (-1.0, 0.6), (0.0, -1.0)]
    for (x, y, yaw), (nx, ny, _) in zip(route, route[1:] + route[:1], strict=True):
        assert yaw == pytest.approx(math.atan2(ny - y, nx - x))


def test_loop_of_two_points_goes_back_and_forth() -> None:
    route = loop_route((0.0, 0.0), [(1.0, 0.0), (0.0, 0.0)])
    assert [(x, y) for x, y, _ in route] == [(0.0, 0.0), (1.0, 0.0)]
    assert route[0][2] == pytest.approx(0.0)
    assert abs(route[1][2]) == pytest.approx(math.pi)


def _room(size_m: float = 1.0, res: float = 0.05) -> tuple[np.ndarray, float, tuple[float, float]]:
    """壁（占有）に囲まれた空きの部屋。原点は左下の角のマスの角。"""
    n = round(size_m / res)
    grid = np.zeros((n, n), np.int16)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100
    return grid, res, (0.0, 0.0)


def test_clearance_is_the_distance_to_the_nearest_wall_cell() -> None:
    grid, res, origin = _room()
    # 中央のマスの中心（0.525, 0.525）から、いちばん近い壁のマスの中心（0.025 か 0.975）まで
    assert clearance_at(grid, res, origin, 0.51, 0.51) == pytest.approx(0.45)


def test_unknown_cells_count_as_blocked() -> None:
    grid, res, origin = _room()
    grid[10, 12] = -1  # (0.625, 0.525) のマス
    assert clearance_at(grid, res, origin, 0.51, 0.51) == pytest.approx(0.1)


def test_a_point_on_a_blocked_cell_has_no_clearance() -> None:
    grid, res, origin = _room()
    assert clearance_at(grid, res, origin, 0.01, 0.51) == 0.0


def test_a_point_outside_the_grid_has_no_clearance() -> None:
    grid, res, origin = _room()
    assert clearance_at(grid, res, origin, -0.5, 0.5) == 0.0


def test_points_can_be_given_as_json_pairs() -> None:
    assert parse_points("[[-0.99, 0.61], [0, -1]]") == [(-0.99, 0.61), (0.0, -1.0)]


def test_points_can_be_given_as_a_flat_list() -> None:
    # ROS のパラメータは入れ子の配列を持てないので、[x1, y1, x2, y2] の並びでも受ける
    assert parse_points([-0.99, 0.61, 0.0, -1.0]) == [(-0.99, 0.61), (0.0, -1.0)]
    assert parse_points("[-0.03, -0.3]") == [(-0.03, -0.3)]


def test_no_points() -> None:
    assert parse_points("") == []
    assert parse_points([]) == []
    assert parse_points(None) == []


def test_an_odd_number_of_values_is_an_error() -> None:
    with pytest.raises(ValueError, match="x, y"):
        parse_points([1.0, 2.0, 3.0])
