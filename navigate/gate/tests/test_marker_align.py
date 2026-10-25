import math

import numpy as np
import pytest

from navgate.marker_align import MarkerRef, align_step, wall_slope


def base_view(robot_xy, robot_yaw, world_pts):
    """壁の座標（壁の法線が +x、壁は x = W）の点を、台車の座標へ写す。"""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    d = np.asarray(world_pts, float) - np.asarray([*robot_xy, 0.0])
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)


WALL_YS = np.linspace(-0.4, 0.4, 81)


def wall(x=1.0, ys=WALL_YS, zs=(0.3, 0.4, 0.5)):
    return np.array([[x, y, z] for y in ys for z in zs])


@pytest.mark.parametrize("yaw_deg", [-8.0, 0.0, 3.0])
def test_wall_slope_is_the_robot_yaw_against_the_wall(yaw_deg):
    pts = base_view((0.2, 0.1), math.radians(yaw_deg), wall())
    assert math.degrees(wall_slope(pts, marker_xy=(0.8, 0.0), marker_z=0.4)) == pytest.approx(
        yaw_deg, abs=1e-6
    )


def test_wall_slope_ignores_points_off_the_wall():
    pts = base_view((0.0, 0.0), math.radians(2.0), wall())
    clutter = np.array([[0.4, y, 0.4] for y in np.linspace(-0.1, 0.1, 30)])  # 手前の物
    assert math.degrees(
        wall_slope(np.vstack([pts, clutter]), marker_xy=(1.0, 0.0), marker_z=0.4)
    ) == pytest.approx(2.0, abs=1e-6)


def test_wall_slope_needs_enough_points():
    assert wall_slope(np.zeros((3, 3)), marker_xy=(1.0, 0.0), marker_z=0.4) is None


MARKER_W = np.array([[1.0, 0.05, 0.43]])


def observe(robot_xy, robot_yaw):
    m = base_view(robot_xy, robot_yaw, MARKER_W)[0]
    return (m[0], m[1]), robot_yaw  # 壁の傾き = 壁に対する台車の向き


def test_at_the_reference_nothing_moves():
    mb, th = observe((0.4, -0.02), math.radians(1.0))
    ref = MarkerRef(marker_id=48, marker_xy=mb, wall_yaw=th)
    assert align_step(mb, th, ref) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize(("xy", "yaw_deg"), [((0.37, 0.01), -4.0), ((0.43, -0.05), 6.0), ((0.4, -0.02), 1.0)])
def test_the_step_returns_the_robot_to_the_reference(xy, yaw_deg):
    ref_xy, ref_yaw = (0.4, -0.02), math.radians(1.0)
    mb0, th0 = observe(ref_xy, ref_yaw)
    ref = MarkerRef(marker_id=48, marker_xy=mb0, wall_yaw=th0)
    yaw = math.radians(yaw_deg)
    mb, th = observe(xy, yaw)
    dth, fwd, left = align_step(mb, th, ref)
    # 回ってから、回ったあとの向きで前と左へ動く
    new_yaw = yaw + dth
    c, s = math.cos(new_yaw), math.sin(new_yaw)
    new_xy = (xy[0] + c * fwd - s * left, xy[1] + s * fwd + c * left)
    assert new_yaw == pytest.approx(ref_yaw, abs=1e-12)
    assert new_xy == pytest.approx(ref_xy, abs=1e-12)
