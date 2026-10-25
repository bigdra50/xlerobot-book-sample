"""teleop_xr（PyRoki）向けの両腕 SO-101 ロボット定義。

teleop_xr 同梱の so101.py（単腕、right だけ）を両腕にしたもの。
リポジトリに同梱した XLeRobot の URDF（dual_urdf.SHIPPED_URDF）を読む。
"""

from __future__ import annotations

from typing import Any, override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy
from teleop_xr.ik.robot import BaseRobot, Cost

from ..contracts import SIDES, ik_joint_name
from .dual_urdf import SHIPPED_URDF, prepare_shipped_urdf

# 同梱 URDF にある、IK で解かない関節。ベースの仮想関節はスティックで、head は HMD で直接動かす
_NON_ARM_JOINTS = ("base_x", "base_y", "base_theta", "head_motor_1", "head_motor_2")


class SO101DualArmRobot(BaseRobot):
    """XLeRobot の両腕（SO-101 ×2）。left / right の手先を 5 自由度ずつ解く。

    gripper は IK から外す（トリガーで直接決める）。
    """

    def __init__(
        self,
        *,
        joint_limits_rad: dict[str, tuple[float, float]] | None = None,
        ee_link: str = "gripper_link",
    ) -> None:
        super().__init__()
        urdf = self._load_default_urdf()

        # 単腕の参照実装と同じ手当て。gripper を fixed にして IK の変数から外す。
        # 同梱 URDF のベースと head も同じく外す
        for name in [ik_joint_name(side, "gripper") for side in SIDES] + list(_NON_ARM_JOINTS):
            if name in urdf.joint_map:
                urdf.joint_map[name].type = "fixed"
                urdf.joint_map[name].mimic = None
        urdf._update_actuated_joints()

        # キャリブレーション実測の可動域は URDF の limit より狭い。
        # pk.Robot.from_urdf が limit を読む前に上書きしておく
        for name, (lower, upper) in (joint_limits_rad or {}).items():
            if name not in urdf.joint_map:
                raise KeyError(f"joint_limits_rad に URDF に無い関節名: {name}")
            if not lower < upper:
                raise ValueError(f"{name} の下限が上限以上: {lower} >= {upper}")
            urdf.joint_map[name].limit.lower = lower
            urdf.joint_map[name].limit.upper = upper

        self.robot = pk.Robot.from_urdf(urdf)

        # 手先 link が無ければ名前の間違いなので、最後の link へ黙って落とさない
        self.ee_link_idx: dict[str, int] = {}
        for side in SIDES:
            link_name = f"{side}_{ee_link}"
            if link_name not in self.robot.links.names:
                raise KeyError(f"手先 link が URDF に無い: {link_name}")
            self.ee_link_idx[side] = self.robot.links.names.index(link_name)
        self._ee_indices = jnp.array([self.ee_link_idx[side] for side in SIDES], dtype=jnp.int32)

    # --- URDF ------------------------------------------------------------------

    @override
    def _load_default_urdf(self) -> yourdfpy.URDF:
        # 同梱 URDF を読む。ネットワークは要らない
        xml = prepare_shipped_urdf(
            SHIPPED_URDF.read_text(encoding="utf-8"), mesh_dir=str(SHIPPED_URDF.parent)
        )
        return self._load_urdf(xml)

    # --- BaseRobot -----------------------------------------------------------------

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        # 同梱 URDF の root（base_footprint）は ROS 規約（X 前、Y 左、Z 上）で置いてあるので回さない
        return jaxlie.SO3.identity()

    @property
    @override
    def supported_frames(self) -> set[str]:
        return set(SIDES)

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {side: jaxlie.SE3(fk[self.ee_link_idx[side]]) for side in SIDES}

    @override
    def get_default_config(self) -> jax.Array:
        return jnp.zeros(self.robot.joints.num_actuated_joints)

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        JointVar = self.robot.joint_var_cls
        joint_var = JointVar(0)

        # 目標が両方 None（warmup）でもコストが空にならないよう、rest_cost は常に付ける
        rest_pose = q_current if q_current is not None else self.get_default_config()
        costs: list[Cost] = [pk.costs.rest_cost(joint_var, rest_pose=rest_pose, weight=1.0)]

        costs.append(pk.costs.manipulability_cost(self.robot, joint_var, self._ee_indices, weight=0.001))

        # 5 自由度なので 6 次元の姿勢は満たせない。位置を優先し、姿勢の重みは弱くする
        for side, target in (("left", target_L), ("right", target_R)):
            if target is None:
                continue
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    joint_var,
                    target,
                    jnp.array(self.ee_link_idx[side], dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=1.0,
                )
            )

        costs.append(pk.costs.limit_cost(self.robot, joint_var, weight=100.0))
        return costs
