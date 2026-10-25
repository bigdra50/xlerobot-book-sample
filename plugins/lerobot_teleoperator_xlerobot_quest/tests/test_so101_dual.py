"""SO101DualArmRobot が同梱の URDF を読み、両腕を左右対称に前へ向けて組むこと。

teleop_xr / pyroki / jax が要る（marker: ik）。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ik

pytest.importorskip("teleop_xr.ik.solver")
jaxlie = pytest.importorskip("jaxlie")

from lerobot_teleoperator_xlerobot_quest.contracts import IK_JOINTS, ik_joint_name
from lerobot_teleoperator_xlerobot_quest.ik.so101_dual import SO101DualArmRobot

EXPECTED_ACTUATED = [ik_joint_name("left", j) for j in IK_JOINTS] + [
    ik_joint_name("right", j) for j in IK_JOINTS
]


def _translation(pose: jaxlie.SE3) -> tuple[float, float, float]:
    return tuple(float(v) for v in pose.translation())


@pytest.fixture(scope="module")
def shipped_robot() -> SO101DualArmRobot:
    # 引数なしは、リポジトリに同梱した urdf/ を読む（ネットワークは要らない）
    return SO101DualArmRobot()


def test_shipped_urdf_solves_only_the_ten_arm_joints(shipped_robot: SO101DualArmRobot) -> None:
    # ベースの仮想関節、head、gripper は fixed にして IK の変数から外す
    assert shipped_robot.actuated_joint_names == EXPECTED_ACTUATED


def test_shipped_urdf_zero_pose_is_mirrored_and_forward(shipped_robot: SO101DualArmRobot) -> None:
    fk = shipped_robot.forward_kinematics(shipped_robot.get_default_config())
    xl, yl, zl = _translation(fk["left"])
    xr, yr, zr = _translation(fk["right"])
    assert xl == pytest.approx(xr, abs=1e-5)
    assert zl == pytest.approx(zr, abs=1e-5)
    assert abs(yl + yr) < 1e-3
    assert yl > 0.1 > -0.1 > yr
    # 手先は取り付け点（x=-0.1286）より前にある
    assert xl > 0.0
