import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

zmq = pytest.importorskip("zmq")
pytest.importorskip("lerobot_robot_xlerobot")

from lerobot_robot_xlerobot.config_xlerobot import XLerobotHostConfig
from xlerobot_host import BookXLerobotHost


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_observation_socket_limits_kernel_send_buffer():
    # 送りきれない観測を OS の送信バッファにためると、どの観測も遅れて届く。ためずに捨てる
    cfg = XLerobotHostConfig(port_zmq_cmd=_free_port(), port_zmq_observations=_free_port())
    host = BookXLerobotHost(cfg, sndbuf_kb=256)
    try:
        assert host.zmq_observation_socket.getsockopt(zmq.SNDBUF) == 256 * 1024
        assert host.zmq_observation_socket.getsockopt(zmq.CONFLATE) == 1
        assert host.zmq_cmd_socket.getsockopt(zmq.CONFLATE) == 1
    finally:
        host.disconnect()
