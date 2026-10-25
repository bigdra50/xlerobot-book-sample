"""plugin を import すると、公式の型と本書の client が lerobot の registry に載る。

lerobot v0.6.1 と公式の lerobot_robot_xlerobot が入った環境で走る。
実機も ZMQ の接続も要らない（connect() を呼ばない）。
"""

import importlib

import pytest

pytest.importorskip("lerobot.robots")


def test_registers_official_and_book_types():
    importlib.import_module("lerobot_robot_xlerobot_book")
    from lerobot.robots.config import RobotConfig

    for name in ("xlerobot", "xlerobot_client", "xlerobot_book_client"):
        assert RobotConfig.get_choice_class(name) is not None, name


def test_client_exposes_camera_names_for_lerobot_record():
    """lerobot-record（v0.6.1）はデータセットを作るときに len(robot.cameras) を数える。"""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.utils import make_robot_from_config
    from lerobot_robot_xlerobot_book import XLerobotBookClientConfig

    names = ("head", "left_wrist", "right_wrist")
    cams = {
        n: OpenCVCameraConfig(index_or_path=i, width=640, height=480, fps=30) for i, n in enumerate(names)
    }
    robot = make_robot_from_config(XLerobotBookClientConfig(remote_ip="127.0.0.1", id="test", cameras=cams))
    assert list(robot.cameras) == list(names)


def test_client_robot_type_matches_recorded_dataset_for_resume():
    """lerobot-record はデータセットの robot_type に robot.name を書き、--resume では robot.robot_type と比べる。"""
    from lerobot.robots.utils import make_robot_from_config
    from lerobot_robot_xlerobot_book import XLerobotBookClientConfig

    sut = make_robot_from_config(XLerobotBookClientConfig(remote_ip="127.0.0.1", id="test"))

    assert sut.robot_type == sut.name == "xlerobot_client"


def test_client_sends_tensor_actions_from_lerobot_replay():
    """lerobot-replay はデータセットの action を Tensor のまま send_action へ渡す。公式の client はそれを JSON にできない。"""
    import json

    import torch
    from lerobot.robots.utils import make_robot_from_config
    from lerobot_robot_xlerobot_book import XLerobotBookClientConfig

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def send_string(self, s):
            self.sent.append(s)

    sut = make_robot_from_config(XLerobotBookClientConfig(remote_ip="127.0.0.1", id="test"))
    sut._is_connected = True  # connect() は ZMQ を開くので、送り先だけを差し替える
    sut.zmq_cmd_socket = FakeSocket()

    sut.send_action({"left_arm_shoulder_pan.pos": torch.tensor(12.5), "x.vel": torch.tensor(0.0)})

    actual = json.loads(sut.zmq_cmd_socket.sent[0])
    assert actual == {"left_arm_shoulder_pan.pos": 12.5, "x.vel": 0.0}
