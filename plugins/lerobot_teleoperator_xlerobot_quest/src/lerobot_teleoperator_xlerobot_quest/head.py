"""HMD の向きの変化を head の pan / tilt に足す。

絶対角ではなく rebase 時点からの差分にする。HMD の yaw は起動時の向きが 0 になり、
head の 0 とは無関係なので、絶対角で結ぶと engage のたびに首が跳ぶ。
"""

from __future__ import annotations

from .contracts import HeadPose


def wrap_deg(deg: float) -> float:
    """±180 に畳む。境界の +180 は -180 になる。"""
    return (deg + 180.0) % 360.0 - 180.0


class HeadFollower:
    """head_motor_1 が pan（yaw）、head_motor_2 が tilt（pitch）。

    モーターの回転方向が HMD の yaw / pitch と逆なら、pan_sign / tilt_sign に -1 を渡す。
    可動域の clamp は呼び出し側がする。
    """

    def __init__(self, *, pan_sign: float = 1.0, tilt_sign: float = 1.0) -> None:
        self._pan_sign = pan_sign
        self._tilt_sign = tilt_sign
        self._ref: tuple[float, float, float, float] | None = None  # (yaw0, pitch0, pan0, tilt0)
        # rebase 前や姿勢ロスト中に返す値。初期は原点
        self._last = (0.0, 0.0)

    def rebase(self, head: HeadPose, pan_deg: float, tilt_deg: float) -> None:
        """現在の HMD の向きと head の現在角を基準にする。

        HMD の姿勢が無いときは基準を捨てて現在角を保持する。
        古い基準のまま追従すると pan0 / tilt0 の差の分だけ首が跳ぶため。
        """
        self._last = (pan_deg, tilt_deg)
        self._ref = (head.yaw_deg, head.pitch_deg, pan_deg, tilt_deg) if head.has_pose else None

    def update(self, head: HeadPose) -> tuple[float, float]:
        if self._ref is None or not head.has_pose:
            return self._last
        yaw0, pitch0, pan0, tilt0 = self._ref
        pan = pan0 + self._pan_sign * wrap_deg(head.yaw_deg - yaw0)
        # pitch は ±90 に収まるので差は ±180 を超えず、畳む必要が無い
        tilt = tilt0 + self._tilt_sign * (head.pitch_deg - pitch0)
        self._last = (pan, tilt)
        return self._last
