import pytest

from navgate.arm_client import (
    Lagging,
    clamp_deg,
    deg_to_norm,
    norm_to_deg,
    parse_targets,
    ramp,
    resolve,
)

CAL = {
    "left_arm_elbow_flex": {"range_min": 1000, "range_max": 3048},
    "left_arm_gripper": {"range_min": 2000, "range_max": 3024},
    "head_motor_1": {"range_min": 1000, "range_max": 3000},
}


def test_body_joints_map_minus_100_to_100_onto_the_calibration_range_in_degrees():
    # 範囲 2048 の半分 1024 は 90 度
    assert norm_to_deg("left_arm_elbow_flex", 100.0, CAL) == pytest.approx(90.0)
    assert norm_to_deg("left_arm_elbow_flex", -50.0, CAL) == pytest.approx(-45.0)
    assert deg_to_norm("left_arm_elbow_flex", 45.0, CAL) == pytest.approx(50.0)


def test_the_gripper_maps_0_to_100_onto_its_range_centered_at_zero_degrees():
    assert norm_to_deg("left_arm_gripper", 50.0, CAL) == pytest.approx(0.0)
    assert norm_to_deg("left_arm_gripper", 0.0, CAL) == pytest.approx(-45.0)
    assert deg_to_norm("left_arm_gripper", 45.0, CAL) == pytest.approx(100.0)


def test_short_names_mean_the_left_arm_and_the_head():
    assert resolve("elbow") == "left_arm_elbow_flex"
    assert resolve("head_pan") == "head_motor_1"
    assert resolve("head_tilt") == "head_motor_2"
    assert resolve("right_arm_gripper") == "right_arm_gripper"
    with pytest.raises(ValueError):
        resolve("knee")


def test_targets_can_be_absolute_or_relative():
    now = {"left_arm_elbow_flex": 10.0, "head_motor_1": -3.0}
    assert parse_targets(["elbow+=5", "head_pan=20"], now) == {
        "left_arm_elbow_flex": 15.0,
        "head_motor_1": 20.0,
    }
    assert parse_targets(["elbow-=12.5"], now) == {"left_arm_elbow_flex": -2.5}


def test_a_malformed_target_is_rejected():
    with pytest.raises(ValueError):
        parse_targets(["elbow~5"], {"left_arm_elbow_flex": 0.0})


def test_targets_are_clamped_inside_the_calibration_range_with_a_margin():
    assert clamp_deg("left_arm_elbow_flex", 200.0, CAL, margin_deg=3.0) == pytest.approx(87.0)
    assert clamp_deg("left_arm_elbow_flex", -200.0, CAL, margin_deg=3.0) == pytest.approx(-87.0)
    assert clamp_deg("left_arm_elbow_flex", 10.0, CAL, margin_deg=3.0) == 10.0


def test_ramp_moves_all_joints_together_at_the_speed_of_the_largest_move():
    steps = ramp({"a": 0.0, "b": 0.0}, {"a": 20.0, "b": 5.0}, speed_dps=20.0, rate_hz=10.0)
    assert len(steps) == 10
    assert steps[0] == pytest.approx({"a": 2.0, "b": 0.5})
    assert steps[-1] == {"a": 20.0, "b": 5.0}


def test_no_move_means_one_step_at_the_target():
    assert ramp({"a": 1.0}, {"a": 1.0}, speed_dps=20.0, rate_hz=10.0) == [{"a": 1.0}]


def test_lagging_trips_only_after_repeated_large_errors():
    lag = Lagging(limit_deg=10.0, count=3)
    assert lag.check({"a": 5.0}, {"a": 0.0}) is None
    assert lag.check({"a": 15.0}, {"a": 0.0}) is None
    assert lag.check({"a": 15.0}, {"a": 0.0}) is None
    assert lag.check({"a": 15.0}, {"a": 0.0}) == "a"


def test_one_small_error_resets_the_count():
    lag = Lagging(limit_deg=10.0, count=3)
    lag.check({"a": 15.0}, {"a": 0.0})
    lag.check({"a": 15.0}, {"a": 0.0})
    assert lag.check({"a": 1.0}, {"a": 0.0}) is None
    assert lag.check({"a": 15.0}, {"a": 0.0}) is None
