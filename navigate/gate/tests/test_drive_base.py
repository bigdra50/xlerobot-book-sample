import sys
import types
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

# drive_base.py は navigate/gate にある単体のスクリプト。main が import する lerobot は偽物に差し替える
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_base

ZERO = dict.fromkeys(drive_base.WHEEL_IDS, 0)


class _FakeBus:
    """FeetechMotorsBus の代わり。呼ばれた操作を順に記録する。"""

    def __init__(self, ids: Iterable[int]) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._ids = tuple(ids)

    def connect(self, handshake: bool = True) -> None:
        self.calls.append(("connect", handshake))

    def broadcast_ping(self) -> dict[int, int]:
        self.calls.append(("broadcast_ping",))
        return dict.fromkeys(self._ids, 777)

    def disable_torque(self) -> None:
        self.calls.append(("disable_torque",))

    def enable_torque(self) -> None:
        self.calls.append(("enable_torque",))

    def write(self, data_name: str, motor: str, value: int, *, normalize: bool = True) -> None:
        self.calls.append(("write", data_name, motor, value))

    def sync_write(self, data_name: str, values: Mapping[str, int], *, num_retry: int = 0) -> None:
        self.calls.append(("sync_write", data_name, dict(values), num_retry))

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))


def _fake_lerobot(monkeypatch: pytest.MonkeyPatch, ids: Iterable[int], *args: str) -> _FakeBus:
    """lerobot を偽物に差し替え、drive_base.main が開くことになるバスを返す。sleep も記録に変える。"""
    bus = _FakeBus(ids)
    motors = types.ModuleType("lerobot.motors")
    motors.Motor = lambda *_: None
    motors.MotorNormMode = types.SimpleNamespace(RANGE_M100_100=None)
    feetech = types.ModuleType("lerobot.motors.feetech")
    feetech.FeetechMotorsBus = lambda **_: bus
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.motors", motors)
    monkeypatch.setitem(sys.modules, "lerobot.motors.feetech", feetech)
    monkeypatch.setattr(drive_base.time, "sleep", lambda seconds: bus.calls.append(("sleep", seconds)))
    monkeypatch.setattr(sys, "argv", ["drive_base.py", "--port", "/dev/fake", *args])
    return bus


def test_main_writes_nothing_when_ids_7_8_9_are_not_all_there(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # bus1（左腕 ID1-6、head ID7-8）を渡した場合。head へ速度も脱力も書かない
    bus = _fake_lerobot(monkeypatch, range(1, 9), "--move", "forward")
    assert drive_base.main() == 1
    assert bus.calls == [("connect", False), ("broadcast_ping",), ("disconnect", False)]
    assert "ID 7,8,9 が揃わない" in capsys.readouterr().out


def test_main_drives_then_writes_zero_velocity_and_releases_torque(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _fake_lerobot(monkeypatch, range(1, 10), "--move", "forward", "--speed", "0.05", "--duration", "1")
    assert drive_base.main() == 0
    assert bus.calls == [
        ("connect", False),
        ("broadcast_ping",),
        ("disable_torque",),
        *[("write", "Operating_Mode", name, 1) for name in drive_base.WHEEL_IDS],
        ("enable_torque",),
        ("sync_write", "Goal_Velocity", drive_base.body_to_wheel_raw(0.05, 0.0, 0.0), 0),
        ("sleep", 1.0),
        ("sync_write", "Goal_Velocity", ZERO, 5),
        ("disconnect", True),
    ]


def test_main_stops_and_releases_torque_on_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _fake_lerobot(monkeypatch, range(1, 10), "--move", "turn-left")

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(drive_base.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        drive_base.main()
    assert bus.calls[-2:] == [("sync_write", "Goal_Velocity", ZERO, 5), ("disconnect", True)]
