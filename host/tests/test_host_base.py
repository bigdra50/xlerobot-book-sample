import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from host_base import BasePort, hold_goal_at_present

WHEELS = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]


class FakeBus:
    def __init__(self, present: dict[str, int] | None = None, fail_reads: int = 0) -> None:
        self.present = present or {}
        self.writes: list[tuple[str, dict]] = []
        self.fail_reads = fail_reads

    def sync_read(self, reg, motors, normalize=True, num_retry=0):
        if self.fail_reads:
            self.fail_reads -= 1
            raise ConnectionError("no status packet")
        assert normalize is False
        if reg == "Present_Position":
            return {m: self.present.get(m, 0) for m in motors}
        return dict.fromkeys(motors, 0)

    def sync_write(self, reg, values, normalize=True, num_retry=0):
        self.writes.append((reg, dict(values)))


def test_goal_is_set_to_the_present_position_before_torque():
    bus = FakeBus({"left_arm_elbow_flex": 3115, "head_motor_1": 2220})
    hold_goal_at_present(bus, ["left_arm_elbow_flex", "head_motor_1"])
    assert bus.writes == [("Goal_Position", {"left_arm_elbow_flex": 3115, "head_motor_1": 2220})]


def _port() -> BasePort:
    return BasePort("127.0.0.1:0", max_linear=0.2, max_angular=45.0, max_current=120, rate_hz=30.0)


def _connect(port: BasePort) -> socket.socket:
    cli = socket.create_connection(("127.0.0.1", port.bound_port), timeout=1.0)
    for _ in range(50):
        if port.poll(time.monotonic()) is not None:
            break
        time.sleep(0.01)
    return cli


def test_when_the_client_leaves_the_wheels_get_zero_and_the_port_is_free_again():
    port = _port()
    cli = _connect(port)
    cli.close()
    bus = FakeBus()
    for _ in range(50):
        if port.poll(time.monotonic(), bus=bus) is None:
            break
        time.sleep(0.01)
    assert ("Goal_Velocity", dict.fromkeys(WHEELS, 0)) in bus.writes
    assert port.poll(time.monotonic()) is None
    port.close(bus)


def test_a_guard_trip_writes_zero_to_the_wheels():
    port = _port()
    cli = _connect(port)
    cli.sendall(b'{"vx": 0.1}\n')
    time.sleep(0.05)
    t0 = time.monotonic()
    port.poll(t0)
    bus = FakeBus()
    bus.sync_read = lambda reg, motors, normalize=True, num_retry=0: (  # type: ignore[method-assign]
        dict.fromkeys(motors, 200) if reg == "Present_Current" else dict.fromkeys(motors, 0)
    )
    for _ in range(12):
        port.report(bus, WHEELS, t0 + 1.0)
    assert ("Goal_Velocity", dict.fromkeys(WHEELS, 0)) in bus.writes
    port.close(bus)
