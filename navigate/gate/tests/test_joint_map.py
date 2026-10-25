import math

import pytest

from navgate.joint_map import HOST_TO_URDF, to_urdf

CAL = {
    name: {"range_min": 1000, "range_max": 3048}  # 半分の幅が 90°
    for name in HOST_TO_URDF
}


def test_every_host_joint_has_a_urdf_joint():
    names = {
        f"{side}_arm_{j}"
        for side in ("left", "right")
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    }
    assert names | {"head_motor_1", "head_motor_2"} == set(HOST_TO_URDF)
    urdf = {m.urdf for m in HOST_TO_URDF.values()}
    assert len(urdf) == 14
    assert {"Rotation_2", "Pitch_2", "Jaw", "head_pan_joint", "head_tilt_joint"} <= urdf


def test_left_arm_uses_the_second_chain_and_the_reference_angle():
    q = to_urdf({"left_arm_shoulder_lift": 0.0}, CAL)
    assert q == {"Pitch_2": pytest.approx(math.radians(84.7))}


def test_head_pan_is_mirrored():
    """host の head_motor_1 は正が右回り、URDF の head_pan_joint は左回りが正。"""
    q = to_urdf({"head_motor_1": 50.0}, CAL)  # 45° 右
    assert q["head_pan_joint"] == pytest.approx(math.radians(-45.0))


def test_unknown_names_are_ignored():
    assert to_urdf({"x.vel": 0.1}, CAL) == {}
