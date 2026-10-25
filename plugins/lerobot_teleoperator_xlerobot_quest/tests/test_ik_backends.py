"""IK backend の選択と、解析ソルバが関節ゼロの姿勢へ往復すること。

実機も Quest も teleop_xr も要らない。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from lerobot_teleoperator_xlerobot_quest.config_xlerobot_vr import XLeRobotVRConfig
from lerobot_teleoperator_xlerobot_quest.contracts import IK_JOINTS
from lerobot_teleoperator_xlerobot_quest.ik.analytic import (
    ANALYTIC_JOINTS,
    AnalyticIKController,
    ClosedFormIK,
)

_ZERO = {name: 0.0 for name in IK_JOINTS}


def _finite(joints: dict[str, float]) -> None:
    assert set(joints) == set(IK_JOINTS)
    for name, angle in joints.items():
        assert math.isfinite(angle), name


class _Pose:
    def __init__(self, position: tuple[float, float, float], quat: tuple[float, float, float, float]) -> None:
        self.position = {"x": position[0], "y": position[1], "z": position[2]}
        self.orientation = {"w": quat[0], "x": quat[1], "y": quat[2], "z": quat[3]}


class _Device:
    def __init__(self, hand: str, pose: _Pose) -> None:
        self.role = "controller"
        self.handedness = hand
        self.gripPose = pose


class _State:
    def __init__(
        self,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        pose = _Pose(position, quat)
        self.devices = [_Device("left", pose), _Device("right", pose)]


# --- 選択 -----------------------------------------------------------------------


def test_draccus_reads_ik_backend() -> None:
    # lerobot-teleoperate では同じ項目が --teleop.ik_backend= になる。
    # draccus 0.10 は Literal を読めないので、フィールドは str Enum。
    import draccus

    cfg = draccus.parse(
        config_class=XLeRobotVRConfig,
        args=["--ik_backend=closed_form"],
        exit_on_error=False,
    )
    assert cfg.ik_backend == "closed_form"


def test_default_ik_backend_is_closed_form() -> None:
    # 本書のコマンドは --teleop.ik_backend を書かない。起動のたびに JIT を待たない closed_form を既定にする
    import draccus

    cfg = draccus.parse(config_class=XLeRobotVRConfig, args=[], exit_on_error=False)
    assert cfg.ik_backend == "closed_form"


# --- closed_form ----------------------------------------------------------------


def test_closed_form_zero_pose_is_finite_and_near_zero() -> None:
    solver = ClosedFormIK()
    back = solver.inverse(solver.forward(_ZERO), _ZERO)
    _finite(back)
    for name, angle in back.items():
        assert angle == pytest.approx(0.0, abs=1e-6), name


def test_closed_form_controller_holds_zero_then_follows() -> None:
    solver = ClosedFormIK()
    controller = AnalyticIKController(solver)
    q0 = np.zeros(len(ANALYTIC_JOINTS))
    held = controller.step(_State((0.2, 0.1, 0.4)), q0)
    assert held == pytest.approx(q0)

    moved = controller.step(_State((0.22, 0.1, 0.4)), held)
    assert np.isfinite(moved).all()
    left = {joint: math.degrees(float(moved[i])) for i, joint in enumerate(IK_JOINTS)}
    home = solver.forward(_ZERO)
    reached = solver.forward(left)
    assert reached.position_m[0] == pytest.approx(home.position_m[0] + 0.02, abs=1e-4)
    assert reached.position_m[1] == pytest.approx(home.position_m[1], abs=1e-4)
