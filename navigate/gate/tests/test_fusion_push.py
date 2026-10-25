import math

import pytest

from navgate.fusion import FusionParams, Pose2D, arkit_base_delta, fuse_step, is_pushed

P = FusionParams()
LENS = (0.128, 0.085)
MOUNT_YAW = 0.0524


def lens_pose(base: Pose2D) -> tuple[float, float, float]:
    """台車の中心の姿勢から、ARKit が返すレンズの位置と yaw を作る。"""
    th = base.theta + MOUNT_YAW
    c, s = math.cos(base.theta), math.sin(base.theta)
    return base.x + c * LENS[0] - s * LENS[1], base.y + s * LENS[0] + c * LENS[1], th


def test_pure_translation_of_the_base_is_recovered_in_the_body_frame():
    b0, b1 = Pose2D(1.0, 2.0, 0.7), Pose2D(1.0 + 0.1 * math.cos(0.7), 2.0 + 0.1 * math.sin(0.7), 0.7)
    x0, y0, a0 = lens_pose(b0)
    x1, y1, a1 = lens_pose(b1)
    assert arkit_base_delta((x0, y0, a0), (x1, y1, a1), LENS, MOUNT_YAW) == pytest.approx(
        (0.1, 0.0, 0.0), abs=1e-9
    )


def test_rotation_in_place_moves_the_lens_but_not_the_base():
    b0, b1 = Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.3)
    dx, dy, dth = arkit_base_delta(lens_pose(b0), lens_pose(b1), LENS, MOUNT_YAW)
    assert (dx, dy) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert dth == pytest.approx(0.3)


def test_the_arkit_world_may_be_rotated_against_odom():
    """ARKit の world の向きは odom と違ってよい。台車の座標で見た変位は同じになる。"""
    b0 = Pose2D(0.0, 0.0, 0.0)
    b1 = fuse_step(b0, 0.05, 0.02, 0.1, None, P).pose
    rot = 1.2

    def rotated(p):
        x, y, a = lens_pose(p)
        c, s = math.cos(rot), math.sin(rot)
        return c * x - s * y, s * x + c * y, a + rot

    assert arkit_base_delta(rotated(b0), rotated(b1), LENS, MOUNT_YAW) == pytest.approx(
        (0.05, 0.02, 0.1), abs=1e-9
    )


@pytest.mark.parametrize(
    ("wheel", "arkit", "pushed"),
    [
        ((0.0, 0.0, 0.0), (0.002, 0.0, 0.0), True),  # 車輪は止まり、ARKit は 0.06 m/s で動いた
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.004), True),  # 押されて 0.12 rad/s で回った
        ((0.0, 0.0, 0.0), (0.0002, 0.0, 0.0), False),  # ARKit の揺れ（0.3 mm 前後）
        ((0.003, 0.0, 0.0), (0.0033, 0.0, 0.0), False),  # 車輪で走っている
    ],
)
def test_is_pushed_when_the_wheels_stand_still_and_arkit_moves(wheel, arkit, pushed):
    assert is_pushed(wheel, arkit, dt=1 / 30, params=P) is pushed


def test_a_push_takes_the_arkit_motion():
    start = Pose2D(0.0, 0.0, math.pi / 2)
    step = fuse_step(start, 0.0, 0.0, 0.0, 0.0, P, arkit_delta=(0.002, -0.001, 0.003), dt=1 / 30)
    assert step.pushed
    assert (step.pose.x, step.pose.y) == pytest.approx((0.001, 0.002), abs=1e-5)
    assert step.pose.theta == pytest.approx(math.pi / 2 + 0.003)


def test_driving_still_takes_the_translation_from_the_wheels():
    start = Pose2D()
    step = fuse_step(start, 0.003, 0.0, 0.0, 0.0, P, arkit_delta=(0.0033, 0.0, 0.0), dt=1 / 30)
    assert not step.pushed
    assert step.pose.x == pytest.approx(0.003)
