import math
import sys
from pathlib import Path

import pytest

from navgate.fusion import Pose2D
from navgate.kinematics import unwrap_steps, wheel_steps_to_body
from navgate.wheel_odom import WheelOdometry

# 車輪の式の正本は navigate/gate/drive_base.py（公式 _body_to_wheel_raw と同じ定数）。navgate の逆変換がそれと合うことを確かめる
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_base


def official_steps(vx: float, vy: float, wz_dps: float) -> dict[str, float]:
    """1 秒ぶんの車輪の回転（step）を、ID の文字列で引ける形にする。"""
    raw = drive_base.body_to_wheel_raw(vx, vy, wz_dps)
    return {str(drive_base.WHEEL_IDS[name]): float(v) for name, v in raw.items()}


@pytest.mark.parametrize(
    ("vx", "vy", "wz"), [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 20.0), (0.05, -0.03, 10.0)]
)
def test_inverse_of_the_official_wheel_formula(vx, vy, wz):
    dx, dy, dth = wheel_steps_to_body(official_steps(vx, vy, wz))
    assert dx == pytest.approx(vx, abs=2e-4)
    assert dy == pytest.approx(vy, abs=2e-4)
    assert math.degrees(dth) == pytest.approx(wz, abs=0.05)


def test_yaw_scale_changes_only_the_rotation():
    steps = official_steps(0.05, 0.02, 15.0)
    dx0, dy0, dth0 = wheel_steps_to_body(steps)
    dx1, dy1, dth1 = wheel_steps_to_body(steps, yaw_scale=1.01)
    assert (dx1, dy1) == pytest.approx((dx0, dy0))
    assert dth1 == pytest.approx(1.01 * dth0)


def test_unwrap_takes_the_short_way_across_the_seam():
    assert unwrap_steps(4090, 5) == 11
    assert unwrap_steps(5, 4090) == -11


def test_first_reading_only_sets_the_reference():
    odo = WheelOdometry()
    assert odo.update({"7": 100, "8": 200, "9": 300}) is None
    assert odo.pose == Pose2D()


def test_driving_forward_one_second():
    odo = WheelOdometry()
    odo.update({"7": 0, "8": 0, "9": 0})
    steps = official_steps(0.1, 0.0, 0.0)
    odo.update({k: int(v) % 4096 for k, v in steps.items()})
    assert odo.pose.x == pytest.approx(0.1, abs=2e-4)
    assert odo.pose.y == pytest.approx(0.0, abs=2e-4)


def test_turning_a_full_circle_is_not_wrapped():
    odo = WheelOdometry()
    pos = {"7": 0, "8": 0, "9": 0}
    odo.update(pos)
    # 1 回の読みのあいだの回転が半周（2048 step）を超えると向きを取り違えるので、45° ずつ読む
    eighth = official_steps(0.0, 0.0, 45.0)
    for _ in range(8):
        pos = {k: int(pos[k] + eighth[k]) % 4096 for k in pos}
        odo.update(pos)
    assert math.degrees(odo.pose.theta) == pytest.approx(360.0, abs=0.2)


def test_readings_across_the_encoder_seam():
    odo = WheelOdometry()
    odo.update({"7": 4000, "8": 4000, "9": 4000})
    delta = odo.update({"7": 104, "8": 104, "9": 104})  # 3 輪とも +200 step（その場の左旋回）
    assert delta is not None
    assert delta[0] == pytest.approx(0.0, abs=1e-9)
    assert delta[2] > 0
