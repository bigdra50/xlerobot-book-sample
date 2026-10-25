"""drive_square.py（台車を四角に 1 周させるスクリプト）の仕様。

純粋な部分（車体と 3 輪の変換、エンコーダの巻き戻し、デッドレコニング、走行計画）と、
指令どおりに回る擬似エンコーダで回した制御ループを確かめる。実機のバスには触れない。
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest

# drive_square.py は navgate の外に置いた単体のスクリプトなので、置き場所を import の探索先へ足す
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drive_square import (
    PERIOD_S,
    WHEEL_IDS,
    Pose2D,
    Segment,
    SimClock,
    SimWheels,
    body_to_wheel_raw,
    closure_error,
    drive,
    dry_run,
    integrate_pose,
    main,
    plan_square,
    run_on_bus,
    unwrap_steps,
    wheel_steps_to_body,
)

ZERO = dict.fromkeys(WHEEL_IDS, 0)


def _square() -> list[Segment]:
    return plan_square(side_m=2.0, sides=4, speed_mps=0.1, turn_dps=20.0)


def test_forward_command_leaves_back_wheel_id8_at_zero() -> None:
    # 後輪 ID8 の駆動方向は横向きなので、純粋な前進では回らない
    raw = body_to_wheel_raw(0.1, 0.0, 0.0)
    assert raw["base_back_wheel"] == 0
    assert raw["base_left_wheel"] == -raw["base_right_wheel"] != 0


@pytest.mark.parametrize(
    ("x_mps", "y_mps", "theta_dps"),
    [
        (0.1, 0.0, 0.0),
        (0.0, 0.1, 0.0),
        (0.0, 0.0, 20.0),
        (-0.1, 0.05, -15.0),
    ],
)
def test_wheel_steps_to_body_inverts_body_to_wheel_raw(x_mps: float, y_mps: float, theta_dps: float) -> None:
    # raw は steps/s なので、1 秒ぶんの差分として戻すと 1 秒ぶんの車体の変位になる。
    # 許容差は、raw を整数へ丸めた誤差（1 輪あたり 0.5 step）から見積もった
    dx, dy, dtheta = wheel_steps_to_body(body_to_wheel_raw(x_mps, y_mps, theta_dps))
    assert dx == pytest.approx(x_mps, abs=1e-4)
    assert dy == pytest.approx(y_mps, abs=1e-4)
    assert math.degrees(dtheta) == pytest.approx(theta_dps, abs=0.05)


def test_unwrap_steps_crosses_the_seam_between_4095_and_0_by_the_short_way() -> None:
    assert unwrap_steps(4090, 5) == 11
    assert unwrap_steps(5, 4090) == -11
    assert unwrap_steps(100, 300) == 200


@pytest.mark.parametrize("pieces", [1, 40])
def test_dead_reckoning_forward_2m_left_90_forward_2m(pieces: int) -> None:
    # 1 回で足しても、制御ループのように細かく刻んで足しても、同じ所へ着く
    pose = Pose2D()
    for dx, dtheta in [(2.0, 0.0), (0.0, math.pi / 2), (2.0, 0.0)]:
        for _ in range(pieces):
            pose = integrate_pose(pose, dx / pieces, 0.0, dtheta / pieces)
    assert pose.x == pytest.approx(2.0)
    assert pose.y == pytest.approx(2.0)
    assert pose.theta == pytest.approx(math.pi / 2)


def test_plan_square_is_four_sides_and_four_left_turns_that_restore_heading() -> None:
    plan = _square()
    assert [s.kind for s in plan] == ["straight", "turn"] * 4
    for side in plan[0::2]:
        # カメラ（iPad）が向いている台車の前、つまり +x を先頭に進む
        assert (side.target, side.vx_mps, side.vy_mps, side.omega_dps) == (2.0, 0.1, 0.0, 0.0)
    for turn in plan[1::2]:
        # 左（上から見て反時計回り）へ 90°
        assert (turn.target, turn.vx_mps, turn.vy_mps, turn.omega_dps) == (90.0, 0.0, 0.0, 20.0)

    # 計画どおりに進めば、始点へ同じ向きで戻る
    pose = Pose2D()
    for segment in plan:
        if segment.kind == "straight":
            pose = integrate_pose(pose, segment.target, 0.0, 0.0)
        else:
            pose = integrate_pose(pose, 0.0, 0.0, math.radians(segment.target))
    distance_m, heading_deg = closure_error(pose)
    assert distance_m == pytest.approx(0.0, abs=1e-9)
    assert heading_deg == pytest.approx(0.0, abs=1e-9)


def test_plan_square_saturates_speed_at_0_2_mps_and_turn_rate_at_30_dps() -> None:
    side, turn = plan_square(side_m=2.0, sides=1, speed_mps=0.5, turn_dps=90.0)
    assert side.vx_mps == 0.2
    assert turn.omega_dps == 30.0


@pytest.mark.parametrize(
    ("side_m", "sides", "speed_mps", "turn_dps"),
    [
        (0.0, 4, 0.1, 20.0),
        (2.0, 0, 0.1, 20.0),
        (2.0, 4, 0.0, 20.0),
        (2.0, 4, 0.1, -20.0),
        # 無限大の辺は時間の上限も無限大になり、止まらなくなる
        (math.inf, 4, 0.1, 20.0),
        (2.0, 4, math.nan, 20.0),
    ],
)
def test_plan_square_rejects_non_positive_or_non_finite_arguments(
    side_m: float, sides: int, speed_mps: float, turn_dps: float
) -> None:
    with pytest.raises(ValueError):
        plan_square(side_m=side_m, sides=sides, speed_mps=speed_mps, turn_dps=turn_dps)


def test_segment_time_limit_is_target_over_rate_times_1_5_plus_1_second() -> None:
    side, turn = plan_square(side_m=2.0, sides=1, speed_mps=0.1, turn_dps=20.0)
    assert side.time_limit_s == pytest.approx(2.0 / 0.1 * 1.5 + 1.0)
    assert turn.time_limit_s == pytest.approx(90.0 / 20.0 * 1.5 + 1.0)


def test_dry_run_closes_the_square_with_near_zero_wheel_odometry_error() -> None:
    result = dry_run(_square())
    assert result.error is None
    assert len(result.segments) == 8
    for side in result.segments[0::2]:
        assert side.progress == pytest.approx(2.0, abs=0.01)
    for turn in result.segments[1::2]:
        assert turn.progress == pytest.approx(90.0, abs=1.0)
    distance_m, heading_deg = closure_error(result.pose)
    assert distance_m < 0.01
    assert abs(heading_deg) < 0.5
    assert result.path_length_m == pytest.approx(8.0, abs=0.05)


class _StuckWheels:
    """指令を受けても回らない車輪。エンコーダの読みは変わらない。"""

    def __init__(self) -> None:
        self.writes: list[dict[str, int]] = []

    def write_velocity(self, raw: Mapping[str, int]) -> None:
        self.writes.append(dict(raw))

    def stop(self) -> None:
        self.writes.append(dict(ZERO))

    def read_steps(self) -> dict[str, int]:
        return dict.fromkeys(WHEEL_IDS, 1000)


def test_drive_stops_and_fails_when_a_segment_runs_past_its_time_limit() -> None:
    plan = _square()
    wheels = _StuckWheels()
    result = drive(wheels, SimClock(), plan)
    assert result.error is not None
    assert "時間の上限" in result.error
    assert len(result.segments) == 1  # 最初の辺で止め、残りの計画へ進まない
    assert result.segments[0].elapsed_s == pytest.approx(plan[0].time_limit_s, abs=2 * PERIOD_S)
    assert wheels.writes[-1] == ZERO


class _WheelsWithFailingReads:
    """指令どおりに回るが、決めた回の読みで ConnectionError（lerobot の sync_read が投げる型）を出す。"""

    def __init__(self, clock: SimClock, failing: Callable[[int], bool]) -> None:
        self._inner = SimWheels(clock)
        self._failing = failing
        self._reads = 0
        self.failed_reads = 0
        self.writes: list[dict[str, int]] = []

    def write_velocity(self, raw: Mapping[str, int]) -> None:
        self.writes.append(dict(raw))
        self._inner.write_velocity(raw)

    def stop(self) -> None:
        self.writes.append(dict(ZERO))
        self._inner.stop()

    def read_steps(self) -> dict[str, int]:
        self._reads += 1
        if self._failing(self._reads):
            self.failed_reads += 1
            raise ConnectionError("Failed to sync read 'Present_Position'")
        return self._inner.read_steps()


def test_drive_stops_after_five_consecutive_read_failures() -> None:
    clock = SimClock()
    wheels = _WheelsWithFailingReads(clock, failing=lambda n: n > 10)
    result = drive(wheels, clock, _square())
    assert result.error is not None
    assert "5 回続けて" in result.error
    assert wheels.failed_reads == 5
    assert wheels.writes[-1] == ZERO


def test_drive_rides_through_isolated_read_failures() -> None:
    # 失敗は続けて数える。間に 1 回でも読めれば数え直す
    clock = SimClock()
    wheels = _WheelsWithFailingReads(clock, failing=lambda n: n % 3 == 0)
    result = drive(wheels, clock, _square())
    assert result.error is None
    assert len(result.segments) == 8
    assert wheels.failed_reads > 5


class _FakeBus:
    """FeetechMotorsBus の代わり。呼ばれた操作を順に記録し、Goal_Velocity を擬似エンコーダへ渡す。"""

    def __init__(self, clock: SimClock, ids: Iterable[int], interrupt_on_read: int | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._ids = tuple(ids)
        self._wheels = SimWheels(clock)
        self._reads = 0
        self._interrupt_on_read = interrupt_on_read

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
        self._wheels.write_velocity(values)

    def sync_read(self, data_name: str, motors: list[str], *, normalize: bool = True) -> dict[str, int]:
        self.calls.append(("sync_read", data_name, normalize))
        self._reads += 1
        if self._reads == self._interrupt_on_read:
            raise KeyboardInterrupt
        return self._wheels.read_steps()

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))


def test_run_on_bus_writes_nothing_when_ids_7_8_9_are_not_all_there() -> None:
    # bus1（左腕 ID1-6、head ID7-8）を指定した場合。head へ速度も脱力も書かない
    clock = SimClock()
    bus = _FakeBus(clock, ids=range(1, 9))
    assert run_on_bus(bus, _square(), clock) is None
    assert bus.calls == [("broadcast_ping",), ("disconnect", False)]


def test_run_on_bus_sets_velocity_mode_then_ends_with_zero_velocity_and_torque_off() -> None:
    clock = SimClock()
    bus = _FakeBus(clock, ids=range(1, 10))
    result = run_on_bus(bus, plan_square(side_m=0.5, sides=1, speed_mps=0.1, turn_dps=20.0), clock)
    assert result is not None
    assert result.error is None
    # 速度モードへの切り替えは脱力中に行い、トルクを入れる前に速度を 0 にしておく
    setup = bus.calls[bus.calls.index(("disable_torque",)) : bus.calls.index(("enable_torque",))]
    assert [c for c in setup if c[0] == "write"] == [
        ("write", "Operating_Mode", name, 1) for name in WHEEL_IDS
    ]
    assert ("sync_write", "Goal_Velocity", ZERO, 5) in setup
    # 校正を渡していないバスでは Present_Position を正規化できないので、生の値で読む
    assert ("sync_read", "Present_Position", False) in bus.calls
    assert bus.calls[-2:] == [("sync_write", "Goal_Velocity", ZERO, 5), ("disconnect", True)]


def test_run_on_bus_stops_and_releases_torque_on_ctrl_c() -> None:
    clock = SimClock()
    bus = _FakeBus(clock, ids=range(1, 10), interrupt_on_read=20)
    with pytest.raises(KeyboardInterrupt):
        run_on_bus(bus, _square(), clock)
    assert bus.calls[-2:] == [("sync_write", "Goal_Velocity", ZERO, 5), ("disconnect", True)]


def test_main_dry_run_prints_plan_each_segment_and_closure(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--dry-run", "--distance", "0.5", "--sides", "1"]) == 0
    out = capsys.readouterr().out
    assert "直進" in out
    assert "旋回" in out
    assert "閉じ誤差" in out


def test_main_needs_port_unless_dry_run() -> None:
    with pytest.raises(SystemExit):
        main([])
