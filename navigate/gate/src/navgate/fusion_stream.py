"""車輪の姿勢と ARKit の yaw が、別々の遅れで届いても、時刻の順に融合する。

ROS 2 のノード（xlerobot_nav の odom_fusion）から使う。ROS には依存しない。
車輪の値は数 ms で届くが、ARKit の姿勢は中継を通って 50〜100 ms 遅れて届く。
車輪の区間ごとに、その区間を ARKit のサンプルが覆うまで待ち、待ちきれなければ車輪だけで進める。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from navgate.fusion import FusionParams, Pose2D, arkit_base_delta, fuse_step


def body_delta(start: Pose2D, end: Pose2D) -> tuple[float, float, float]:
    """fuse_step の逆。2 つの姿勢から、区間の中ほどの向きで見た車体座標の変位を求める。"""
    dtheta = end.theta - start.theta
    mid = start.theta + dtheta / 2.0
    c, s = math.cos(mid), math.sin(mid)
    ex, ey = end.x - start.x, end.y - start.y
    return c * ex + s * ey, -s * ex + c * ey, dtheta


@dataclass(frozen=True)
class _Arkit:
    t: float
    yaw: float
    ok: bool
    epoch: int
    x: float | None = None
    y: float | None = None


class FusionStream:
    def __init__(
        self,
        params: FusionParams,
        max_delay_s: float = 0.3,
        lens_xy: tuple[float, float] = (0.0, 0.0),
        mount_yaw: float = 0.0,
    ) -> None:
        """lens_xy と mount_yaw は、台車の中心から見た ARKit のレンズの位置と向き（URDF の取り付け）。"""
        self.params = params
        self.max_delay_s = max_delay_s
        self.lens_xy = lens_xy
        self.mount_yaw = mount_yaw
        self.last_pushed = False
        self.pose: Pose2D | None = None
        self._wheels: list[tuple[float, Pose2D]] = []
        self._arkit: list[_Arkit] = []

    def feed_wheel(self, t: float, pose: Pose2D) -> None:
        """車輪だけで積み上げた姿勢（/wheel/odom）を渡す。最初の姿勢が、融合した姿勢の始点になる。"""
        if self.pose is None:
            self.pose = pose
        self._wheels.append((t, pose))

    def feed_arkit(
        self, t: float, yaw: float, ok: bool, epoch: int, x: float | None = None, y: float | None = None
    ) -> None:
        """ARKit の yaw [rad] と、あればレンズの位置 [m] を渡す。ok はトラッキングが normal か、epoch は world の原点の番号。

        位置を渡すと、車輪が止まっているのに台車が動いた（押された、滑った）区間を ARKit で積み上げる。
        """
        item = _Arkit(t, yaw, ok, epoch, x, y)
        i = bisect.bisect_right([a.t for a in self._arkit], t)
        self._arkit.insert(i, item)

    def ready(self, now: float) -> list[tuple[float, Pose2D, float]]:
        """融合できた区間の (時刻, 融合した姿勢, ARKit の重み) を、古い順に返す。"""
        out: list[tuple[float, Pose2D, float]] = []
        while len(self._wheels) >= 2:
            (t0, w0), (t1, w1) = self._wheels[0], self._wheels[1]
            dth_a = self._arkit_increment(t0, t1)
            if dth_a == "wait":
                if now - t1 <= self.max_delay_s:
                    break
                dth_a = None
            dx, dy, dth = body_delta(w0, w1)
            assert self.pose is not None
            delta = self._arkit_delta(t0, t1) if dth_a is not None else None
            step = fuse_step(self.pose, dx, dy, dth, dth_a, self.params, arkit_delta=delta, dt=t1 - t0)
            self.pose = step.pose
            self.last_pushed = step.pushed
            out.append((t1, step.pose, step.arkit_weight))
            self._wheels.pop(0)
        self._trim()
        return out

    def _arkit_increment(self, t0: float, t1: float) -> float | None | str:
        times = [a.t for a in self._arkit]
        if not times or times[-1] < t1:
            return "wait"
        i0 = bisect.bisect_right(times, t0) - 1
        if i0 < 0:
            return None  # t0 より前のサンプルが無い（ARKit が後から始まった）
        i1 = bisect.bisect_left(times, t1)
        span = self._arkit[i0 : i1 + 1]
        if not all(a.ok for a in span) or len({a.epoch for a in span}) != 1:
            return None
        ts = np.array([a.t for a in span])
        yaw = np.unwrap(np.array([a.yaw for a in span]))
        return float(np.interp(t1, ts, yaw) - np.interp(t0, ts, yaw))

    def _arkit_delta(self, t0: float, t1: float) -> tuple[float, float, float] | None:
        """t0 から t1 までの、ARKit で見た台車の中心の変位。位置の無いサンプルが混じれば None。"""
        times = [a.t for a in self._arkit]
        i0 = bisect.bisect_right(times, t0) - 1
        i1 = bisect.bisect_left(times, t1)
        span = self._arkit[max(i0, 0) : i1 + 1]
        if len(span) < 2 or any(a.x is None or a.y is None for a in span):
            return None
        ts = np.array([a.t for a in span])
        yaw = np.unwrap(np.array([a.yaw for a in span]))
        xs = np.array([a.x for a in span], dtype=float)
        ys = np.array([a.y for a in span], dtype=float)

        def at(t: float) -> tuple[float, float, float]:
            return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys)), float(np.interp(t, ts, yaw))

        return arkit_base_delta(at(t0), at(t1), self.lens_xy, self.mount_yaw)

    def _trim(self) -> None:
        if not self._wheels or len(self._arkit) < 2:
            return
        t_next = self._wheels[0][0]
        keep_from = max(0, bisect.bisect_right([a.t for a in self._arkit], t_next - 1.0) - 1)
        del self._arkit[:keep_from]
