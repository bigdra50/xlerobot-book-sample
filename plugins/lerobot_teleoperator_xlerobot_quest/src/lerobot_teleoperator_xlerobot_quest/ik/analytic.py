"""腕ごとに独立して解く IK。PyRoki の両腕同時の最適化は使わない。

- closed_form: 同梱した so101_chain。肩の平面選択と、平面内の 3R。肘の枝を明示する

ソルバの入力は、その腕の base での手先位置（m）と pitch / roll（度）で、出力は 5 関節の度。
AnalyticIKController が、engage の瞬間の手先と手を覚え、以後は手の変位を手先へ足して左右を別々に解く。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..contracts import IK_JOINTS, SIDES, ik_joint_name
from .hand_axes import orientation_offsets
from .vendor import so101_chain as chain

# 手のひねりと wrist_roll は回る向きが逆。
# 実機の closed_form で、gripper が手と逆へ回った。
# 移植元の lekiwi-vr-teleop も、roll の向きは腕の付け方で変わるので実機を見て決めるとしている（roll_gain）
ROLL_SIGN = -1.0

# q の並び。左の 5 関節、続いて右の 5 関節（rad）
ANALYTIC_JOINTS = [ik_joint_name(side, joint) for side in SIDES for joint in IK_JOINTS]


@dataclass(frozen=True)
class ArmPose:
    """1 本の腕の base での目標。位置は m、角度は度。"""

    position_m: tuple[float, float, float]
    pitch_deg: float
    roll_deg: float


class ArmIK(Protocol):
    name: str

    def forward(self, joints_deg: dict[str, float]) -> ArmPose: ...

    def inverse(self, target: ArmPose, current_deg: dict[str, float] | None = None) -> dict[str, float]: ...


def _joints(values: dict[str, float]) -> dict[str, float]:
    return {name: float(values.get(name, 0.0)) for name in IK_JOINTS}


class ClosedFormIK:
    """so101_chain.inverse。肘は上げる枝で解く。

    ゼロ関節の手先を解くと、関節はゼロへ戻る。
    """

    name = "closed_form"

    def forward(self, joints_deg: dict[str, float]) -> ArmPose:
        joints = _joints(joints_deg)
        tool = chain.forward(joints)["tool"]
        pitch = chain.tool_pitch(joints["shoulder_lift"], joints["elbow_flex"], joints["wrist_flex"])
        return ArmPose(
            position_m=tuple(float(v) for v in tool), pitch_deg=pitch, roll_deg=joints["wrist_roll"]
        )

    def inverse(self, target: ArmPose, current_deg: dict[str, float] | None = None) -> dict[str, float]:
        # current を渡すと、URDF の可動域の外にいる関節を engage で引き戻さない
        pose = chain.inverse(
            target.position_m,
            target.pitch_deg,
            target.roll_deg,
            elbow_up=True,
            current=_joints(current_deg) if current_deg is not None else None,
        )
        return pose.as_dict()


@dataclass(frozen=True)
class _Grip:
    """手の grip 姿勢。位置は m、quaternion は xyzw。"""

    position: np.ndarray
    quat_xyzw: np.ndarray


@dataclass(frozen=True)
class _Latch:
    """engage したときの手先と手。"""

    home: ArmPose
    grip: _Grip


def grip_poses(state: Any) -> dict[str, _Grip]:
    """teleop_xr の XRState から左右の grip 姿勢を取る。無い手は結果に入れない。

    role と handedness は str の Enum なので、文字列のまま比べてキーにできる。
    """
    found: dict[str, _Grip] = {}
    for device in state.devices:
        pose = device.gripPose
        if device.role != "controller" or pose is None or device.handedness not in SIDES:
            continue
        p, o = pose.position, pose.orientation
        found[device.handedness] = _Grip(
            position=np.array([p["x"], p["y"], p["z"]], dtype=float),
            quat_xyzw=np.array([o["x"], o["y"], o["z"], o["w"]], dtype=float),
        )
    return found


class AnalyticIKController:
    """IKController.step と同じ呼び出しで、腕ごとの解析 IK を返す。

    step は追従中にだけ呼ぶ。両手の grip が揃っていないときは、q をそのまま返す。
    追従を始めた最初の呼び出しは基準を取るだけで、関節は変えない。
    次の呼び出しから、手の変位を解く。
    """

    def __init__(self, solver: ArmIK) -> None:
        self.solver = solver
        self._active = False
        self._latch: dict[str, _Latch] = {}
        self._index = {name: i for i, name in enumerate(ANALYTIC_JOINTS)}

    def reset(self) -> None:
        self._active = False
        self._latch.clear()

    def step(self, state: Any, q_current: np.ndarray) -> np.ndarray:
        q = np.asarray(q_current, dtype=float).copy()
        if q.shape != (len(ANALYTIC_JOINTS),):
            raise ValueError(f"q の長さが {len(ANALYTIC_JOINTS)} ではない: {q.shape}")
        poses = grip_poses(state)
        if any(side not in poses for side in SIDES):
            self.reset()
            return q
        if not self._active:
            self._latch = {side: self._take_latch(q, side, poses[side]) for side in SIDES}
            self._active = True
            return q
        for side in SIDES:
            self._write_side(q, side, poses[side])
        return q

    def _joints_deg(self, q: np.ndarray, side: str) -> dict[str, float]:
        return {joint: math.degrees(float(q[self._index[ik_joint_name(side, joint)]])) for joint in IK_JOINTS}

    def _take_latch(self, q: np.ndarray, side: str, grip: _Grip) -> _Latch:
        return _Latch(home=self.solver.forward(self._joints_deg(q, side)), grip=grip)

    def _write_side(self, q: np.ndarray, side: str, grip: _Grip) -> None:
        latch = self._latch[side]
        tool = np.asarray(latch.home.position_m) + (grip.position - latch.grip.position)
        pitch_off, roll_off = orientation_offsets(latch.grip.quat_xyzw, grip.quat_xyzw)
        target = ArmPose(
            position_m=(float(tool[0]), float(tool[1]), float(tool[2])),
            pitch_deg=latch.home.pitch_deg + pitch_off,
            roll_deg=latch.home.roll_deg + ROLL_SIGN * roll_off,
        )
        solved = self.solver.inverse(target, self._joints_deg(q, side))
        for joint in IK_JOINTS:
            q[self._index[ik_joint_name(side, joint)]] = math.radians(solved[joint])
