import math

import pytest

from navgate.fusion import FusionParams, Pose2D, arkit_weight, fuse_step

P = FusionParams()


def _integrate(steps) -> Pose2D:
    """(dx, dy, dtheta_wheel, dtheta_arkit) の列を順に積み上げた姿勢。"""
    pose = Pose2D()
    for dx, dy, dth_w, dth_a in steps:
        pose = fuse_step(pose, dx, dy, dth_w, dth_a, P).pose
    return pose


def test_turning_in_place_takes_the_wheel_yaw():
    step = fuse_step(Pose2D(), 0.0, 0.0, 0.10, 0.102, P)
    assert step.arkit_weight == 0.0
    assert step.dtheta == pytest.approx(0.10)


def test_translating_takes_the_arkit_yaw():
    step = fuse_step(Pose2D(), 0.0, 0.01, 0.0, 0.002, P)
    assert step.arkit_weight == 1.0
    assert step.dtheta == pytest.approx(0.002)


def test_without_arkit_the_wheels_are_used():
    step = fuse_step(Pose2D(), 0.01, 0.0, 0.003, None, P)
    assert step.arkit_weight == 0.0
    assert step.dtheta == pytest.approx(0.003)


def test_standing_still_ignores_arkit_jitter():
    step = fuse_step(Pose2D(), 0.0, 0.0, 0.0, 0.001, P)
    assert step.dtheta == 0.0
    assert step.pose == Pose2D()


def test_mixed_motion_is_weighted_by_the_error_variances():
    params = FusionParams(
        arkit_rot_scale_err=0.02, wheel_yaw_err_per_m_fwd=0.08, wheel_yaw_err_per_m_side=0.22
    )
    var_wheel = (0.08 * 0.01) ** 2
    var_arkit = (0.02 * 0.02) ** 2
    assert arkit_weight(0.01, 0.0, 0.02, params) == pytest.approx(var_wheel / (var_wheel + var_arkit))


def test_sideways_motion_trusts_arkit_more_than_forward_motion():
    assert arkit_weight(0.0, 0.01, 0.02, P) > arkit_weight(0.01, 0.0, 0.02, P)


def test_translation_follows_the_fused_heading():
    after_turn = fuse_step(Pose2D(), 0.0, 0.0, math.pi / 2, None, P).pose
    moved = fuse_step(after_turn, 1.0, 0.0, 0.0, None, P).pose
    assert moved.x == pytest.approx(0.0, abs=1e-12)
    assert moved.y == pytest.approx(1.0)


def test_a_step_is_integrated_at_its_mid_heading():
    moved = fuse_step(Pose2D(), 1.0, 0.0, 0.2, None, P).pose
    assert (moved.x, moved.y) == pytest.approx((math.cos(0.1), math.sin(0.1)))
    assert moved.theta == pytest.approx(0.2)


def test_heading_is_accumulated_without_wrapping():
    steps = [(0.0, 0.0, math.pi / 4, None)] * 8
    assert _integrate(steps).theta == pytest.approx(2 * math.pi)


def test_strafe_rotation_that_the_wheels_miss_is_taken_from_arkit():
    # 横へ 0.4 m のあいだに台車が実際に 5° 回り、車輪の値には表れない場面
    steps = [(0.0, 0.02, 0.0, math.radians(5.0) / 20)] * 20
    fused = _integrate(steps)
    wheels = _integrate([(dx, dy, dth, None) for dx, dy, dth, _ in steps])
    assert math.degrees(fused.theta) == pytest.approx(5.0)
    assert wheels.theta == 0.0


def test_arkit_overcounting_a_turn_is_not_taken():
    # 180° の旋回で、車輪 176°、ARKit 179.4° の場面
    steps = [(0.0, 0.0, math.radians(176.0) / 40, math.radians(179.4) / 40)] * 40
    assert math.degrees(_integrate(steps).theta) == pytest.approx(176.0)
