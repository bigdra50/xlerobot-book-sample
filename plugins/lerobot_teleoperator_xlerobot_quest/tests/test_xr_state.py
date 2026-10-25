"""xr_state: teleop_xr の XRState（model_dump した dict）から VRSnapshot を作る。"""

from __future__ import annotations

import math

import pytest
from lerobot_teleoperator_xlerobot_quest.contracts import Controller, VRSnapshot
from lerobot_teleoperator_xlerobot_quest.xr_state import snapshot_from_xr_state


def _button(pressed: bool = False, value: float | None = None, touched: bool = False) -> dict:
    return {"pressed": pressed, "touched": touched, "value": float(pressed) if value is None else value}


def _pose(w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"w": w, "x": x, "y": y, "z": z}}


def _controller(handedness: str, buttons: list[dict], axes: list[float], pose: dict | None) -> dict:
    return {
        "role": "controller",
        "handedness": handedness,
        "gripPose": pose,
        "gamepad": {"buttons": buttons, "axes": axes},
    }


def _quat_z(deg: float) -> dict:
    h = math.radians(deg) / 2
    return _pose(w=math.cos(h), z=math.sin(h))


def _full_state() -> dict:
    left_buttons = [
        _button(value=0.25, touched=True),  # 0 trigger
        _button(pressed=True),  # 1 squeeze
        _button(),  # 2 touchpad
        _button(),  # 3 thumbstick
        _button(),  # 4 X
        _button(pressed=True),  # 5 Y
    ]
    right_buttons = [
        _button(value=1.0),
        _button(),
        _button(),
        _button(),
        _button(pressed=True),  # 4 A
        _button(),
    ]
    return {
        "timestamp_unix_ms": 1.0e12,
        "devices": [
            _controller("left", left_buttons, [0.0, 0.0, -0.5, 0.75], _pose()),
            _controller("right", right_buttons, [0.1, 0.2, 0.3, -0.4], _pose()),
            {"role": "head", "pose": _quat_z(90.0)},
        ],
    }


def test_full_state_maps_every_field():
    snap = snapshot_from_xr_state(_full_state())
    assert isinstance(snap, VRSnapshot)

    assert snap.left == Controller(
        trigger=0.25, squeeze=True, primary=False, secondary=True, stick_x=-0.5, stick_y=0.75, has_pose=True
    )
    assert snap.right == Controller(
        trigger=1.0, squeeze=False, primary=True, stick_x=0.3, stick_y=-0.4, has_pose=True
    )
    assert snap.head.has_pose is True
    assert snap.head.yaw_deg == pytest.approx(90.0)
    assert snap.head.pitch_deg == pytest.approx(0.0)


@pytest.mark.parametrize("deg", [-135.0, -90.0, -30.0, 0.0, 45.0, 90.0, 170.0])
def test_head_yaw_from_z_rotation(deg):
    state = {"devices": [{"role": "head", "pose": _quat_z(deg)}]}
    snap = snapshot_from_xr_state(state)
    assert snap.head.yaw_deg == pytest.approx(deg)
    assert snap.head.pitch_deg == pytest.approx(0.0, abs=1e-9)
