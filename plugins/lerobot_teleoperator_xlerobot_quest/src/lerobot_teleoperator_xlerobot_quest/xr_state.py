"""teleop_xr の XRState（model_dump した dict）から VRSnapshot を作る。

dict の形は teleop_xr 1.3.5 の messages.py に合わせてある。
teleop_xr 自体は import しない。VR サーバを立てない環境でもこの変換をテストするため。
"""

from __future__ import annotations

import math
from typing import Any

from .contracts import Controller, HeadPose, VRSnapshot

# WebXR xr-standard gamepad のボタン添字
_BTN_TRIGGER = 0
_BTN_SQUEEZE = 1
_BTN_PRIMARY = 4  # A（右）/ X（左）
_BTN_SECONDARY = 5  # B（右）/ Y（左）
# axes は [touchpad_x, touchpad_y, thumbstick_x, thumbstick_y]
_AXIS_STICK_X = 2
_AXIS_STICK_Y = 3


def _button(buttons: list[Any], index: int) -> dict[str, Any]:
    # 配列が短い（機種によってボタン数が違う）ときは「押されていない」扱い
    if index < len(buttons) and isinstance(buttons[index], dict):
        return buttons[index]
    return {}


def _axis(axes: list[Any], index: int) -> float:
    if index < len(axes) and axes[index] is not None:
        return float(axes[index])
    return 0.0


def _controller(device: dict[str, Any]) -> Controller:
    gamepad = device.get("gamepad") or {}
    buttons = gamepad.get("buttons") or []
    axes = gamepad.get("axes") or []
    return Controller(
        # トリガーは押し込み量をグリッパ開度に使うので pressed ではなく value を読む
        trigger=float(_button(buttons, _BTN_TRIGGER).get("value") or 0.0),
        squeeze=bool(_button(buttons, _BTN_SQUEEZE).get("pressed", False)),
        primary=bool(_button(buttons, _BTN_PRIMARY).get("pressed", False)),
        secondary=bool(_button(buttons, _BTN_SECONDARY).get("pressed", False)),
        stick_x=_axis(axes, _AXIS_STICK_X),
        stick_y=_axis(axes, _AXIS_STICK_Y),
        has_pose=device.get("gripPose") is not None,
    )


def yaw_pitch_from_quaternion(w: float, x: float, y: float, z: float) -> tuple[float, float]:
    """ROS 規約（X 前・Y 左・Z 上）の quaternion から yaw（Z 軸回り）と pitch（Y 軸回り）を度で返す。

    teleop_xr が WebXR 座標系から ROS 規約へ変換済みの姿勢を前提にする。
    roll は head に対応する軸が無いので出さない。
    """
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # 丸め誤差で |arg| が 1 を超えると asin が ValueError になるので clamp する
    pitch = math.asin(min(1.0, max(-1.0, 2 * (w * y - z * x))))
    return math.degrees(yaw), math.degrees(pitch)


def _head(device: dict[str, Any]) -> HeadPose:
    pose = device.get("pose")
    if not pose:
        return HeadPose(has_pose=False)
    q = pose.get("orientation") or {}
    yaw, pitch = yaw_pitch_from_quaternion(
        float(q.get("w", 1.0)), float(q.get("x", 0.0)), float(q.get("y", 0.0)), float(q.get("z", 0.0))
    )
    return HeadPose(yaw_deg=yaw, pitch_deg=pitch, has_pose=True)


def snapshot_from_xr_state(state: dict[str, Any]) -> VRSnapshot:
    """欠けたデバイスは既定値（has_pose=False）にする。ヘッドセットの起動直後は devices が空。"""
    left = Controller()
    right = Controller()
    head = HeadPose()
    for device in state.get("devices") or []:
        role = device.get("role")
        if role == "controller":
            hand = device.get("handedness")
            if hand == "left":
                left = _controller(device)
            elif hand == "right":
                right = _controller(device)
        elif role == "head":
            head = _head(device)
    return VRSnapshot(left=left, right=right, head=head)
