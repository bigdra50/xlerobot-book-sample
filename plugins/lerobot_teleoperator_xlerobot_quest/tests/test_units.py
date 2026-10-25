"""units: lerobot 0.6.1 の _normalize / _unnormalize と同じ式で度と正規化値を往復する。"""

from __future__ import annotations

import pytest
from lerobot.motors import MotorCalibration
from lerobot_teleoperator_xlerobot_quest import units


@pytest.fixture
def calibration(calibration_dict) -> dict[str, MotorCalibration]:
    return {name: MotorCalibration(**entry) for name, entry in calibration_dict.items()}


@pytest.fixture
def cal_arm(calibration) -> MotorCalibration:
    """range 1024..3072、drive_mode 0"""
    return calibration["left_arm_shoulder_pan"]


def test_raw_to_deg_matches_lerobot_formula(cal_arm):
    # lerobot: (raw - mid) * 360 / (4096 - 1)
    assert units.raw_to_deg(1024.0, cal_arm) == pytest.approx((1024 - 2048) * 360 / 4095)


@pytest.mark.parametrize("deg", [-80.0, -1.0, 0.0, 45.0, 80.0])
def test_deg_normalized_round_trip(cal_arm, deg):
    assert units.normalized_to_deg(units.deg_to_normalized(deg, cal_arm), cal_arm) == pytest.approx(deg)


def test_to_robot_units_normalized_converts_joints_and_passes_gripper(calibration):
    deg = {
        "left_arm_shoulder_pan": 0.0,
        "left_arm_elbow_flex": units.raw_to_deg(3072, calibration["left_arm_elbow_flex"]),
        "left_arm_gripper": 37.5,
        "head_motor_1": units.raw_to_deg(1024, calibration["head_motor_1"]),
    }
    out = units.to_robot_units(deg, calibration)
    assert set(out) == set(deg)  # キーは入力のまま。".pos" を付けない
    assert out["left_arm_shoulder_pan"] == pytest.approx(0.0)
    assert out["left_arm_elbow_flex"] == pytest.approx(100.0)
    assert out["left_arm_gripper"] == 37.5
    assert out["head_motor_1"] == pytest.approx(-100.0)
