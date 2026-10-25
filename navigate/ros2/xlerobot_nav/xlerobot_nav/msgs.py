"""メッセージとの変換。向きは水平面の yaw だけを扱う。"""

from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, TransformStamped


def to_stamp(t: float) -> Time:
    sec = math.floor(t)
    nanosec = round((t - sec) * 1e9)
    if nanosec >= 1_000_000_000:
        sec, nanosec = sec + 1, nanosec - 1_000_000_000
    return Time(sec=int(sec), nanosec=int(nanosec))


def to_seconds(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def quaternion_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def transform_to_matrix(tf: TransformStamped) -> np.ndarray:
    """TF の変換を 4x4 の同次変換行列にする。"""
    q, t = tf.transform.rotation, tf.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = (t.x, t.y, t.z)
    return T


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Unwrapper:
    """±π で折り返す yaw を、続けて積み上げた値へ直す。"""

    def __init__(self) -> None:
        self.raw: float | None = None
        self.value = 0.0

    def update(self, raw: float) -> float:
        if self.raw is None:
            self.value = raw
        else:
            self.value += wrap(raw - self.raw)
        self.raw = raw
        return self.value
