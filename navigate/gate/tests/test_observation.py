import base64

from navgate.observation import split_observation


def test_cameras_are_decoded_from_base64_jpeg():
    jpeg = b"\xff\xd8fake\xff\xd9"
    obs = {"head": base64.b64encode(jpeg).decode(), "left_arm_gripper.pos": 1.5}
    images, _joints, _base = split_observation(obs)
    assert images == {"head": jpeg}


def test_a_camera_the_host_failed_to_encode_is_skipped():
    images, _, _ = split_observation({"head": "", "left_wrist": base64.b64encode(b"x").decode()})
    assert set(images) == {"left_wrist"}


def test_joints_are_named_without_the_pos_suffix_in_a_stable_order():
    obs = {"right_arm_elbow_flex.pos": 2.0, "head_motor_1.pos": -1.0, "left_arm_elbow_flex.pos": 3.0}
    _, joints, _ = split_observation(obs)
    assert joints == [("head_motor_1", -1.0), ("left_arm_elbow_flex", 3.0), ("right_arm_elbow_flex", 2.0)]


def test_base_velocity_is_returned_in_m_per_s_and_deg_per_s():
    _, _, base = split_observation({"x.vel": 0.1, "y.vel": -0.02, "theta.vel": 5.0})
    assert base == (0.1, -0.02, 5.0)


def test_missing_base_velocity_is_none():
    assert split_observation({})[2] is None
