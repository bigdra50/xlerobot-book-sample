"""base: 左右スティックから台車の速度指令へ。符号は実機で確認する前提。"""

from __future__ import annotations

import pytest
from lerobot_teleoperator_xlerobot_quest.base import stick_to_base_velocity
from lerobot_teleoperator_xlerobot_quest.contracts import BASE_KEYS, Controller

SPEED = {"xy_speed_mps": 0.2, "theta_speed_dps": 60.0, "deadzone": 0.15}


def vel(lx=0.0, ly=0.0, rx=0.0, ry=0.0, **overrides):
    kw = {**SPEED, **overrides}
    return stick_to_base_velocity(
        Controller(stick_x=lx, stick_y=ly), Controller(stick_x=rx, stick_y=ry), **kw
    )


def test_neutral_is_all_zero():
    out = vel()
    assert set(out) == set(BASE_KEYS)
    assert out == {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


def test_forward_gives_positive_x():
    out = vel(ly=-1.0)
    assert out["x.vel"] == pytest.approx(0.2)
    assert out["y.vel"] == pytest.approx(0.0)
    assert out["theta.vel"] == 0.0


def test_right_stick_right_turns_clockwise():
    # theta.vel は左回り正。右に倒すと負
    out = vel(rx=1.0)
    assert out["theta.vel"] == pytest.approx(-60.0)
    assert out["x.vel"] == 0.0 and out["y.vel"] == 0.0
