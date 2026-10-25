"""車輪オドメトリと ARKit の yaw を融合し、台車の 2D の姿勢を積み上げる。

重みは、動きの種類ごとの誤差の大きさから決める。

- その場の旋回では、車輪の回転が正確で、ARKit は回転を 0.35〜2.1 % 多く見積もる
- 直進と平行移動のあいだに台車が回っても、車輪の値には表れない。ARKit はこの回転を捉える
- 並進は車輪から取る。無地の扉へ向かう 0.6 m で、ARKit は 9 % 長く見積もった
- ただし車輪が止まっているのに ARKit が動いたときは、手で押されたか床を滑ったとみなし、並進も回転も ARKit から取る
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    """始点の座標で見た位置 [m] と向き [rad]。向きは巻かずに積み上げる（1 周の旋回を 2π として残すため）。"""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass(frozen=True)
class FusionParams:
    # ARKit の回転の比率の誤差（1σ）。その場の旋回で 0.35〜2.1 % 多く見積もったので、その上限に合わせる
    arkit_rot_scale_err: float = 0.02
    # 車輪の値に表れない回転 [rad/m]。前へ 0.6〜0.9 m 進むあいだに 3〜4° 回った
    wheel_yaw_err_per_m_fwd: float = 0.08
    # 同じく横 [rad/m]。横へ 0.4 m 動くあいだに 2〜8° 回った
    wheel_yaw_err_per_m_side: float = 0.22
    # 車輪の変位がこれより小さい区間は止まっているとみなし、ARKit の揺れを取り込まない
    still_m: float = 1e-4
    still_rad: float = 1e-4
    # 押されたとみなす速さ。止まった ARKit の揺れは 0.3 mm 前後（30 Hz で 0.01 m/s）
    push_min_speed: float = 0.03
    push_min_rate: float = 0.05
    # 車輪がこれより遅ければ止まっているとみなす
    wheel_still_speed: float = 0.005
    wheel_still_rate: float = 0.01


@dataclass(frozen=True)
class Step:
    pose: Pose2D
    dtheta: float
    arkit_weight: float
    pushed: bool = False


def arkit_weight(dx: float, dy: float, dtheta_wheel: float, params: FusionParams) -> float:
    """ARKit の yaw に置く重み（0〜1）。車輪と ARKit の yaw の誤差の分散から、逆分散で決める。"""
    if math.hypot(dx, dy) < params.still_m and abs(dtheta_wheel) < params.still_rad:
        return 0.0
    var_wheel = (params.wheel_yaw_err_per_m_fwd * dx) ** 2 + (params.wheel_yaw_err_per_m_side * dy) ** 2
    var_arkit = (params.arkit_rot_scale_err * dtheta_wheel) ** 2
    return var_wheel / (var_wheel + var_arkit)


def arkit_base_delta(
    lens0: tuple[float, float, float],
    lens1: tuple[float, float, float],
    lens_xy: tuple[float, float],
    mount_yaw: float,
) -> tuple[float, float, float]:
    """ARKit のレンズの 2 つの姿勢 (x, y, yaw) から、台車の中心の変位を、区間の中ほどの向きの車体座標で返す。

    lens_xy は台車の中心から見たレンズの位置、mount_yaw はレンズの向きと台車の向きの差。
    レンズは中心から離れているので、その場で回るだけでも円を描く。その分を差し引く。
    """
    th0, th1 = lens0[2] - mount_yaw, lens1[2] - mount_yaw
    rx, ry = lens_xy

    def arm(th: float) -> tuple[float, float]:
        c, s = math.cos(th), math.sin(th)
        return c * rx - s * ry, s * rx + c * ry

    a0, a1 = arm(th0), arm(th1)
    ex = (lens1[0] - lens0[0]) - (a1[0] - a0[0])
    ey = (lens1[1] - lens0[1]) - (a1[1] - a0[1])
    dth = math.remainder(th1 - th0, math.tau)
    mid = th0 + dth / 2.0
    c, s = math.cos(mid), math.sin(mid)
    return c * ex + s * ey, -s * ex + c * ey, dth


def is_pushed(
    wheel: tuple[float, float, float], arkit: tuple[float, float, float], dt: float, params: FusionParams
) -> bool:
    """車輪は止まっているのに、ARKit で見た台車が動いたか。"""
    if dt <= 0:
        return False
    wheel_still = (
        math.hypot(wheel[0], wheel[1]) / dt < params.wheel_still_speed
        and abs(wheel[2]) / dt < params.wheel_still_rate
    )
    arkit_moving = (
        math.hypot(arkit[0], arkit[1]) / dt >= params.push_min_speed
        or abs(arkit[2]) / dt >= params.push_min_rate
    )
    return wheel_still and arkit_moving


def _advance(pose: Pose2D, dx: float, dy: float, dtheta: float) -> Pose2D:
    mid = pose.theta + dtheta / 2.0
    c, s = math.cos(mid), math.sin(mid)
    return Pose2D(pose.x + c * dx - s * dy, pose.y + s * dx + c * dy, pose.theta + dtheta)


def fuse_step(
    pose: Pose2D,
    dx: float,
    dy: float,
    dtheta_wheel: float,
    dtheta_arkit: float | None,
    params: FusionParams,
    arkit_delta: tuple[float, float, float] | None = None,
    dt: float | None = None,
) -> Step:
    """車体座標の車輪の変位 (dx, dy, dtheta_wheel) と、同じ区間の ARKit の yaw の変化から、次の姿勢を求める。

    arkit_delta（ARKit で見た台車の変位、arkit_base_delta の値）と dt を渡すと、押されたときにそちらを使う。
    """
    if (
        arkit_delta is not None
        and dt is not None
        and is_pushed((dx, dy, dtheta_wheel), arkit_delta, dt, params)
    ):
        return Step(_advance(pose, *arkit_delta), arkit_delta[2], 1.0, pushed=True)
    w = 0.0 if dtheta_arkit is None else arkit_weight(dx, dy, dtheta_wheel, params)
    dtheta = dtheta_wheel if dtheta_arkit is None else dtheta_wheel + w * (dtheta_arkit - dtheta_wheel)
    return Step(_advance(pose, dx, dy, dtheta), dtheta, w)
