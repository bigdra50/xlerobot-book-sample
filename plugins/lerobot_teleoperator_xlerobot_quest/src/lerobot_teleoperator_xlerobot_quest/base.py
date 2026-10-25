"""スティックから台車の速度指令へ。

出力は公式 XLerobot の base action と同じ {"x.vel", "y.vel", "theta.vel"}。
単位は x.vel / y.vel が m/s、theta.vel が deg/s。
"""

from __future__ import annotations

import math

from .contracts import Controller


def _radial_deadzone(x: float, y: float, deadzone: float) -> tuple[float, float]:
    """半径で判定し、deadzone を超えた分だけを 0..1 に再スケールする。

    軸ごとに判定すると斜め入力で片軸だけ死んで向きが歪むので、半径で判定する。
    """
    r = math.hypot(x, y)
    if r <= deadzone:
        return 0.0, 0.0
    scale = min(1.0, (r - deadzone) / (1.0 - deadzone))
    return x / r * scale, y / r * scale


def stick_to_base_velocity(
    left: Controller,
    right: Controller,
    *,
    xy_speed_mps: float,
    theta_speed_dps: float,
    deadzone: float,
) -> dict[str, float]:
    """左スティックで並進、右スティックの x で旋回。

    符号は WebXR の thumbstick 規約（前に倒すと y が負、左に倒すと x が負）と
    ROS 規約（+X 前、+Y 左、+theta 左回り）から決めた。
    """
    if not 0.0 <= deadzone < 1.0:
        raise ValueError(f"deadzone は 0 以上 1 未満（{deadzone}）")
    lx, ly = _radial_deadzone(left.stick_x, left.stick_y, deadzone)
    rx, _ = _radial_deadzone(right.stick_x, right.stick_y, deadzone)
    return {
        "x.vel": -ly * xy_speed_mps,
        "y.vel": -lx * xy_speed_mps,
        "theta.vel": -rx * theta_speed_dps,
    }
