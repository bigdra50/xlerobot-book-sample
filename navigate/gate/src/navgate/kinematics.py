"""XLeRobot のオムニ 3 輪の逆変換。車輪の回転（エンコーダの step）から、車体座標の変位を求める。

定数は navigate/gate/drive_base.py（公式 XLerobot._body_to_wheel_raw と同じ）に合わせてある。
navgate を Pi 側のスクリプトに依存させないため写して持ち、一致はテストで確かめる。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import numpy as np

WHEEL_ORDER: Final[tuple[str, ...]] = ("7", "8", "9")  # base_left_wheel, base_back_wheel, base_right_wheel
MOUNT_ANGLES_DEG: Final[tuple[float, ...]] = (150.0, -90.0, 30.0)
WHEEL_RADIUS_M: Final[float] = 0.05
BASE_RADIUS_M: Final[float] = 0.125
STEPS_PER_TURN: Final[int] = 4096

_INVERSE: Final = np.linalg.inv(
    np.array(
        [[math.cos(math.radians(a)), math.sin(math.radians(a)), BASE_RADIUS_M] for a in MOUNT_ANGLES_DEG]
    )
)


def unwrap_steps(prev: int, now: int) -> int:
    """Present_Position の差分を、0〜4095 の継ぎ目をまたいでも短い向きで返す。"""
    half = STEPS_PER_TURN // 2
    return (now - prev + half) % STEPS_PER_TURN - half


def wheel_steps_to_body(steps: Mapping[str, float], yaw_scale: float = 1.0) -> tuple[float, float, float]:
    """3 輪の回転（step、ID の文字列で引く）を、車体座標の (dx [m], dy [m], dtheta [rad]) へ戻す。

    yaw_scale は旋回の較正の係数で、dtheta だけに掛かる。3 輪の配置が対称なので、
    中心から車輪までの距離を変えることと同じになる。
    """
    arcs = [steps[k] * 2.0 * math.pi / STEPS_PER_TURN * WHEEL_RADIUS_M for k in WHEEL_ORDER]
    dx, dy, dtheta = _INVERSE @ np.array(arcs)
    return float(dx), float(dy), float(dtheta) * yaw_scale
