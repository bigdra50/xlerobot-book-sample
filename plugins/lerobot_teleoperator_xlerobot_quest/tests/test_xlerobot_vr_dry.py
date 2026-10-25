"""Teleoperator を実機もヘッドセットも無しに connect → get_action → disconnect まで通す。

teleop_xr の VR サーバをローカルの空きポートで立て、XR 入力は与えない。
確かめるのは、契約どおりの 17 キーが出ること、基準姿勢が無ければ腕を動かさないこと、
observation_bus から基準を取れば保持指令になること。
どれも IK の backend に依らないので、jax の要らない closed_form で立てる。
"""

from __future__ import annotations

import socket
import time

import pytest

pytest.importorskip("teleop_xr")

from lerobot.motors import MotorCalibration
from lerobot_teleoperator_xlerobot_quest.config_xlerobot_vr import XLeRobotVRConfig
from lerobot_teleoperator_xlerobot_quest.contracts import ACTION_KEYS, BASE_KEYS, MOTOR_NAMES


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def teleop(tmp_path, calibration_dict):
    import draccus

    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    cal = {k: MotorCalibration(**v) for k, v in calibration_dict.items()}
    with open(cal_dir / "dry.json", "w") as f, draccus.config_type("json"):
        draccus.dump(cal, f, indent=2)

    cfg = XLeRobotVRConfig(id="dry", calibration_dir=cal_dir, port=_free_port(), ik_backend="closed_form")
    from lerobot_teleoperator_xlerobot_quest.xlerobot_vr import XLeRobotVR

    t = XLeRobotVR(cfg)
    assert len(t.calibration) == 17
    t.connect()
    deadline = time.monotonic() + 10
    while not t.is_connected and time.monotonic() < deadline:
        time.sleep(0.1)
    assert t.is_connected
    yield t
    t.disconnect()
    assert not t.is_connected


def test_action_features_match_contract(teleop):
    assert list(teleop.action_features) == list(ACTION_KEYS)


def test_without_reference_only_base_is_commanded(teleop):
    try:
        from lerobot_robot_xlerobot_book import observation_bus

        observation_bus._latest = None  # 他のテストの残りを消す
    except ImportError:
        pass
    action = teleop.get_action()
    assert set(action) == set(BASE_KEYS)
    assert all(v == 0.0 for v in action.values())


def test_holds_observed_pose_when_idle(teleop):
    observation_bus = pytest.importorskip("lerobot_robot_xlerobot_book.observation_bus")
    observed = {f"{m}.pos": (50.0 if m.endswith("gripper") else 10.0) for m in MOTOR_NAMES}
    observed.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    observation_bus.publish(observed)

    action = teleop.get_action()
    assert list(action) == list(ACTION_KEYS)
    for m in MOTOR_NAMES:
        assert action[f"{m}.pos"] == pytest.approx(observed[f"{m}.pos"], abs=1e-6), m
    assert all(action[k] == 0.0 for k in BASE_KEYS)
    # 入力が無い間は何度呼んでも同じ指令のまま
    assert teleop.get_action() == action


def test_correction_start_retakes_reference_from_robot(teleop):
    # dagger では補正のあいだしか get_action が呼ばれない。
    # 2 回目の補正の最初の指令が、前回の補正の終わりの姿勢へ向かわないこと
    observation_bus = pytest.importorskip("lerobot_robot_xlerobot_book.observation_bus")
    dagger = pytest.importorskip("lerobot.rollout.strategies.dagger")
    import logging

    def observed(deg: float) -> dict[str, float]:
        obs = {f"{m}.pos": (50.0 if m.endswith("gripper") else deg) for m in MOTOR_NAMES}
        obs.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        return obs

    observation_bus.publish(observed(10.0))
    teleop.get_action()
    moved = observed(30.0)
    observation_bus.publish(moved)

    log = logging.getLogger(dagger.__name__)
    level = log.level
    log.setLevel(logging.INFO)
    try:
        log.info("Phase transition: %s -> %s", "paused", "correcting")
    finally:
        log.setLevel(level)
    action = teleop.get_action()
    for m in MOTOR_NAMES:
        assert action[f"{m}.pos"] == pytest.approx(moved[f"{m}.pos"], abs=1e-6), m
