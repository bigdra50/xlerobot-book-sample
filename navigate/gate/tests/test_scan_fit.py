import math

import numpy as np
import pytest

from navgate.scan_fit import free_hit_ratio

RES = 0.05


def room():
    """3 m x 2 m の部屋。壁の外と棚は、中まで占有で埋まっている（rtabmap の格子と同じ）。"""
    cells = np.full((40, 60), 100, np.int16)
    cells[1:-1, 1:-1] = 0
    cells[5:12, 40:48] = 100
    return cells


FOV = math.radians(60)


def scan_from(cells, x, y, th, n=180, fov=FOV, max_r=3.0):
    pts = []
    for a in np.linspace(-fov / 2, fov / 2, n):
        for r in np.arange(0.05, max_r, RES / 2):
            i, j = int((x + r * math.cos(th + a)) / RES), int((y + r * math.sin(th + a)) / RES)
            if not (0 <= i < cells.shape[1] and 0 <= j < cells.shape[0]) or cells[j, i] > 50:
                pts.append((r * math.cos(a), r * math.sin(a)))
                break
    return np.array(pts)


@pytest.mark.parametrize("th_deg", [0.0, 90.0, -120.0])
def test_at_the_true_pose_no_point_falls_on_free_cells(th_deg):
    cells = room()
    pose = (1.2, 0.9, math.radians(th_deg))
    ratio, center_free = free_hit_ratio(scan_from(cells, *pose), cells, RES, (0.0, 0.0), pose)
    assert ratio < 0.05
    assert center_free


def test_a_wrong_estimate_puts_points_on_free_cells():
    cells = room()
    pts = scan_from(cells, 1.2, 0.9, 0.0)
    ratio, _ = free_hit_ratio(pts, cells, RES, (0.0, 0.0), (0.7, 0.9, 0.0))
    assert ratio > 0.5


def test_an_estimate_inside_an_obstacle_is_flagged():
    cells = room()
    pts = scan_from(cells, 1.2, 0.9, 0.0)
    _, center_free = free_hit_ratio(pts, cells, RES, (0.0, 0.0), (2.2, 0.4, 0.0))
    assert not center_free
