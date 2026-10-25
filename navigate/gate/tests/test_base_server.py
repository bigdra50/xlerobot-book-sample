import json
import math
import socket
import sys
from pathlib import Path

import pytest

# base_server.py は Pi で走らせる単体のスクリプトなので、置き場所を import の探索先へ足す
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base_server import BaseSession, ClientClosed, Guard, Limits, Watchdog, clamp_command, row_line

LIM = Limits(max_linear_mps=0.2, max_angular_dps=45.0)


def test_a_command_inside_the_limits_is_kept():
    assert clamp_command({"vx": 0.1, "vy": -0.05, "wz": 20.0}, LIM) == (0.1, -0.05, 20.0)


def test_linear_speed_is_scaled_down_keeping_its_direction():
    vx, vy, _ = clamp_command({"vx": 0.3, "vy": 0.4, "wz": 0.0}, LIM)
    assert math.hypot(vx, vy) == pytest.approx(0.2)
    assert vy / vx == pytest.approx(0.4 / 0.3)


def test_angular_speed_is_saturated():
    assert clamp_command({"vx": 0.0, "vy": 0.0, "wz": -90.0}, LIM)[2] == -45.0


@pytest.mark.parametrize(
    "msg", [{"vx": "fast"}, {"vx": float("nan")}, {"wz": float("inf")}, {"vx": None}, []]
)
def test_a_malformed_command_means_stop(msg):
    assert clamp_command(msg, LIM) is None


def test_missing_fields_are_zero():
    assert clamp_command({"wz": 10.0}, LIM) == (0.0, 0.0, 10.0)


def test_watchdog_expires_without_commands():
    dog = Watchdog(timeout_s=0.3)
    assert dog.expired(now=10.0)
    dog.feed(now=10.0)
    assert not dog.expired(now=10.25)
    assert dog.expired(now=10.31)


def test_guard_trips_on_three_overcurrent_readings_in_a_row():
    g = Guard(max_current=120, min_ratio=0.3, over_limit=3, slow_limit=5, grace_s=0.4)
    targets = {"7": 500, "8": 500, "9": 500}
    row = {"cur": {"7": 130, "8": 10, "9": 10}, "vel": {"7": 500, "8": 500, "9": 500}}
    assert g.check(row, targets, since_change_s=1.0) is None
    assert g.check(row, targets, since_change_s=1.0) is None
    assert "current" in g.check(row, targets, since_change_s=1.0)


def test_guard_ignores_a_wheel_that_is_still_accelerating():
    g = Guard(max_current=120, min_ratio=0.3, over_limit=3, slow_limit=5, grace_s=0.4)
    targets = {"7": 500, "8": 0, "9": 500}
    row = {"cur": {"7": 10, "8": 10, "9": 10}, "vel": {"7": 0, "8": 0, "9": 0}}
    assert all(g.check(row, targets, since_change_s=0.2) is None for _ in range(10))
    assert all(g.check(row, targets, since_change_s=0.5) is None for _ in range(4))
    assert "7" in g.check(row, targets, since_change_s=0.5)


def test_row_line_is_one_json_line():
    line = row_line(12.5, {"7": 1, "8": 2, "9": 3}, {"7": 0, "8": 0, "9": 0}, {"7": 5, "8": 6, "9": 7})
    assert line.endswith("\n") and line.count("\n") == 1
    assert json.loads(line) == {
        "t": 12.5,
        "pos": {"7": 1, "8": 2, "9": 3},
        "vel": {"7": 0, "8": 0, "9": 0},
        "cur": {"7": 5, "8": 6, "9": 7},
    }


# --- BaseSession: 1 つの接続の指令と読みを扱う。バスには触れず、書く値を返す ---

NAMES = ("base_left_wheel", "base_back_wheel", "base_right_wheel")


def _pair():
    a, b = socket.socketpair()
    b.settimeout(1.0)
    return a, b


def _lines(sock: socket.socket) -> list[dict]:
    sock.settimeout(0.2)
    buf = b""
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    except TimeoutError:
        pass
    return [json.loads(x) for x in buf.decode().splitlines() if x]


def _session(conn: socket.socket) -> BaseSession:
    return BaseSession(conn, LIM, max_current=120, rate_hz=30.0)


def test_session_announces_ready_with_its_rate():
    srv, cli = _pair()
    _session(srv).begin()
    assert _lines(cli) == [{"event": "ready", "rate_hz": 30.0}]


def test_session_targets_are_zero_until_a_command_arrives():
    srv, _cli = _pair()
    s = _session(srv)
    assert s.targets(now=0.0) == dict.fromkeys(NAMES, 0)


def test_session_turns_a_command_into_wheel_targets():
    srv, cli = _pair()
    s = _session(srv)
    cli.sendall(b'{"vx": 0.1, "vy": 0.0, "wz": 0.0}\n')
    t = s.targets(now=1.0)
    assert t["base_back_wheel"] == 0
    assert t["base_left_wheel"] == -t["base_right_wheel"] != 0


def test_session_stops_when_commands_stop_arriving():
    srv, cli = _pair()
    s = _session(srv)
    cli.sendall(b'{"vx": 0.1}\n')
    assert any(s.targets(now=1.0).values())
    assert not any(s.targets(now=1.31).values())


def test_session_raises_when_the_client_closes():
    srv, cli = _pair()
    s = _session(srv)
    cli.close()
    with pytest.raises(ClientClosed):
        s.targets(now=0.0)


def test_session_reports_rows_keyed_by_wheel_id():
    srv, cli = _pair()
    s = _session(srv)
    pos = {"base_left_wheel": 1, "base_back_wheel": 2, "base_right_wheel": 3}
    zero = dict.fromkeys(NAMES, 0)
    assert s.report(5.0, pos, zero, zero, now=0.0) is False
    assert _lines(cli) == [
        {
            "t": 5.0,
            "pos": {"7": 1, "8": 2, "9": 3},
            "vel": {"7": 0, "8": 0, "9": 0},
            "cur": {"7": 0, "8": 0, "9": 0},
        }
    ]


def test_session_latches_a_stop_on_overcurrent_until_a_zero_command():
    srv, cli = _pair()
    s = _session(srv)
    cli.sendall(b'{"vx": 0.1}\n')
    targets = s.targets(now=1.0)
    hot = dict.fromkeys(NAMES, 200)
    tripped = [s.report(0.0, dict.fromkeys(NAMES, 0), targets, hot, now=2.0) for _ in range(3)]
    assert tripped == [False, False, True]
    assert any(line.get("event") == "guard" for line in _lines(cli))
    cli.sendall(b'{"vx": 0.1}\n')
    assert not any(s.targets(now=2.1).values())
    cli.sendall(b'{"vx": 0.0}\n{"vx": 0.1}\n')
    assert any(s.targets(now=2.2).values())
