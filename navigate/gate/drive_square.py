"""台車を四角に 1 周させ、車輪オドメトリの閉じ誤差を出す。

ARKit の姿勢の記録（gate-record）のあいだ、人のテレオペの代わりに台車を走らせる。
Mac で gate-record を先に始めておき、これを Raspberry Pi の venv（install.sh が作る ~/xlerobot-book-env）の Python で走らせる。

drive_base.py と同じく bus2（右腕 ID1-6 + ホイール ID7-9）を直接開き、ホイール 3 台にだけ書く。
車体速度から 3 輪の raw への変換は drive_base.body_to_wheel_raw をそのまま使い、方向の定義も同じにする。
iPad（カメラ）は台車の前を向いているので、カメラを先頭にした走行は +x（drive_base.py の forward）になる。

各辺と各旋回では、指令を書いてから約 20 Hz でエンコーダ（Present_Position）を読み、
積分した距離か向きが目標に届いたら止める。
隅では左（上から見て反時計回り）へ 90° 回るので、四角は最初の進行方向の左側にできる。
4 辺なら、最後の旋回で向きが始点と同じに戻る。

初めて床で走らせるときは、drive_base.py で輪を浮かせて方向を確かめてから、--distance 0.5 --sides 1 で試す。

Usage:
    python drive_square.py --dry-run
    python drive_square.py --port /dev/ttyACM1 --distance 0.5 --sides 1
    python drive_square.py --port /dev/ttyACM1

Exit code: 計画を走り切ったら 0。ID 7-9 が揃わない、ポートが開けない、時間の上限を超えた、
エンコーダを読めない、Ctrl+C のどれかで止めたら 1。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

import numpy as np

# 式と定数を二重に持たないため、同じディレクトリの drive_base.py から import する。
from drive_base import (
    BASE_RADIUS_M,
    MOUNT_ANGLES_DEG,
    STEPS_PER_DEG,
    WHEEL_IDS,
    WHEEL_RADIUS_M,
    body_to_wheel_raw,
)

# Present_Position は 0〜4095 で 1 周する
STEPS_PER_TURN: Final[int] = round(STEPS_PER_DEG * 360.0)

# ARKit のトラッキングを保つための上限。引数がこれを超えても飽和させる
MAX_SPEED_MPS: Final[float] = 0.2
MAX_TURN_DPS: Final[float] = 30.0

PERIOD_S: Final[float] = 0.05
# 止めたあとも読み続け、惰性で進んだぶんを積分へ入れる。区間の境で止まるのは ARKit にもよい
SETTLE_S: Final[float] = 0.5
# 巻き戻しは、読みの間隔で半周（2048 steps）より回らない前提に立つ。上限の速度で最も速い輪は約 0.9 秒で半周する。
# この前提には、読みが 5 回続けて失敗して止めるまでの時間が 0.9 秒より短いことが要る
MAX_READ_FAILURES: Final[int] = 5

_READ_FAILED: Final[str] = f"エンコーダの読みが {MAX_READ_FAILURES} 回続けて失敗した"

# body_to_wheel_raw の変換行列。行が各輪、列が車体の (x [m], y [m], theta [rad])。
# 公式 XLerobot._wheel_raw_to_body と同じく、この行列を解いて 3 輪から車体へ戻す
_WHEEL_MATRIX: Final = np.array(
    [[math.cos(math.radians(a)), math.sin(math.radians(a)), BASE_RADIUS_M] for a in MOUNT_ANGLES_DEG]
)


def _zero_velocity() -> dict[str, int]:
    return dict.fromkeys(WHEEL_IDS, 0)


def unwrap_steps(prev: int, now: int) -> int:
    """Present_Position の差分を、0〜4095 の継ぎ目をまたいでも短い向きで返す。"""
    half = STEPS_PER_TURN // 2
    return (now - prev + half) % STEPS_PER_TURN - half


def wheel_steps_to_body(steps: Mapping[str, float]) -> tuple[float, float, float]:
    """3 輪のエンコーダの差分（steps）を、車体座標の変位 (dx [m], dy [m], dtheta [rad]) へ戻す。"""
    arcs_m = [math.radians(steps[name] / STEPS_PER_DEG) * WHEEL_RADIUS_M for name in WHEEL_IDS]
    dx, dy, dtheta = np.linalg.solve(_WHEEL_MATRIX, arcs_m)
    return float(dx), float(dy), float(dtheta)


@dataclass(frozen=True)
class Pose2D:
    """始点の車体座標で見た位置 [m] と向き [rad]。向きは巻かずに積み上げる。"""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


def integrate_pose(pose: Pose2D, dx: float, dy: float, dtheta: float) -> Pose2D:
    """車体座標の変位を、積分した向きで始点の座標へ足す（デッドレコニング）。"""
    # 区間の中ほどの向きで回す。旋回しながら進んだ区間でも、始めの向きで回すよりずれが小さい
    heading = pose.theta + dtheta / 2.0
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return Pose2D(
        x=pose.x + dx * cos_h - dy * sin_h,
        y=pose.y + dx * sin_h + dy * cos_h,
        theta=pose.theta + dtheta,
    )


def closure_error(pose: Pose2D) -> tuple[float, float]:
    """始点からの距離 [m] と、始点との向きの差 [deg]（-180 より大きく 180 以下）。"""
    heading_deg = math.degrees(pose.theta) % 360.0
    if heading_deg > 180.0:
        heading_deg -= 360.0
    return math.hypot(pose.x, pose.y), heading_deg


@dataclass(frozen=True)
class Segment:
    """走行計画の 1 区間。target は、直進なら m、旋回なら deg。"""

    kind: Literal["straight", "turn"]
    target: float
    vx_mps: float
    vy_mps: float
    omega_dps: float

    @property
    def rate(self) -> float:
        """target の単位で見た、指令どおりの 1 秒あたりの進み（m/s か deg/s）。"""
        if self.kind == "straight":
            return math.hypot(self.vx_mps, self.vy_mps)
        return abs(self.omega_dps)

    @property
    def time_limit_s(self) -> float:
        """目標 / 速度 × 1.5 + 1 秒。滑りや読みの遅れで遅れても、ここまでは待つ。"""
        return self.target / self.rate * 1.5 + 1.0

    @property
    def label(self) -> str:
        return "直進" if self.kind == "straight" else "旋回"

    @property
    def unit(self) -> str:
        return "m" if self.kind == "straight" else "deg"

    def progress(self, dx: float, dy: float, dtheta: float) -> float:
        """車体座標の変位のうち、この区間の指令の向きへ進んだぶん（target の単位）。"""
        if self.kind == "straight":
            return (dx * self.vx_mps + dy * self.vy_mps) / self.rate
        return math.degrees(dtheta) * math.copysign(1.0, self.omega_dps)


def plan_square(side_m: float, sides: int, speed_mps: float, turn_dps: float) -> list[Segment]:
    """1 辺ごとに、カメラの側（前、+x）へ side_m 進んで、左へ 90° 回る計画。

    4 辺なら四角を 1 周し、最後の旋回で向きが始点と同じに戻る。速度と旋回速度は上限で飽和させる。
    """
    for name, value in (
        ("1 辺の長さ", side_m),
        ("辺の数", sides),
        ("速度", speed_mps),
        ("旋回速度", turn_dps),
    ):
        # nan は比較をすり抜け、inf の辺は時間の上限も inf にするので、有限であることも求める
        if not (value > 0 and math.isfinite(value)):
            raise ValueError(f"{name}は正の有限な値にする（{value}）")
    side = Segment("straight", side_m, min(speed_mps, MAX_SPEED_MPS), 0.0, 0.0)
    corner = Segment("turn", 90.0, 0.0, 0.0, min(turn_dps, MAX_TURN_DPS))
    return [side, corner] * sides


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class Wheels(Protocol):
    """3 輪への速度の指令と、エンコーダの読み。read_steps は読めないとき OSError を投げる。"""

    def write_velocity(self, raw: Mapping[str, int]) -> None: ...

    def stop(self) -> None: ...

    def read_steps(self) -> dict[str, int]: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class SimClock:
    """dry-run とテストの時計。sleep は時刻を進めるだけで、実際には待たない。"""

    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds


class SimWheels:
    """指令どおりに回る擬似エンコーダ。滑りも惰性も無い。"""

    def __init__(self, clock: Clock, start: Mapping[str, int] | None = None) -> None:
        self._clock = clock
        self._t = clock.now()
        self._position = {name: float((start or {}).get(name, 0)) for name in WHEEL_IDS}
        self._velocity = _zero_velocity()

    def _advance(self) -> None:
        now = self._clock.now()
        for name in WHEEL_IDS:
            self._position[name] += self._velocity[name] * (now - self._t)
        self._t = now

    def write_velocity(self, raw: Mapping[str, int]) -> None:
        self._advance()
        self._velocity = {name: int(raw[name]) for name in WHEEL_IDS}

    def stop(self) -> None:
        self.write_velocity(_zero_velocity())

    def read_steps(self) -> dict[str, int]:
        self._advance()
        # 実機と同じく、整数へ丸めて 0〜4095 で巻いた値を返す
        return {name: round(position) % STEPS_PER_TURN for name, position in self._position.items()}


class _BusWheels:
    """FeetechMotorsBus（bus2）のホイール 3 台。"""

    def __init__(self, bus: Any) -> None:
        self._bus = bus

    def write_velocity(self, raw: Mapping[str, int]) -> None:
        self._bus.sync_write("Goal_Velocity", dict(raw))

    def stop(self) -> None:
        self._bus.sync_write("Goal_Velocity", _zero_velocity(), num_retry=5)

    def read_steps(self) -> dict[str, int]:
        # 校正を渡していないバスでは Present_Position を正規化できないので、生の値で読む。
        # 失敗は drive が続けて数えるので、ここでは再試行しない
        values = self._bus.sync_read("Present_Position", list(WHEEL_IDS), normalize=False)
        return {name: int(values[name]) for name in WHEEL_IDS}


@dataclass(frozen=True)
class SegmentResult:
    """1 区間の結果。progress は止めたあとの惰性を含み、elapsed_s は指令から停止までの時間。"""

    index: int
    segment: Segment
    progress: float
    elapsed_s: float


@dataclass(frozen=True)
class DriveResult:
    """走行の結果。error が None なら計画を走り切った。pose と経路長は車輪オドメトリの値。"""

    segments: tuple[SegmentResult, ...]
    pose: Pose2D
    path_length_m: float
    error: str | None = None


class _Reader:
    """エンコーダを読み、続けて失敗した回数を数える。"""

    def __init__(self, wheels: Wheels) -> None:
        self._wheels = wheels
        self.failures = 0

    @property
    def gave_up(self) -> bool:
        return self.failures >= MAX_READ_FAILURES

    def read(self) -> dict[str, int] | None:
        try:
            steps = self._wheels.read_steps()
        except OSError:  # lerobot の ConnectionError も pyserial の SerialException も OSError の仲間
            self.failures += 1
            return None
        self.failures = 0
        return steps


class _Odometry:
    """3 輪のエンコーダの読みを、巻き戻し、逆変換、デッドレコニングの順に積み上げる。"""

    def __init__(self, steps: Mapping[str, int]) -> None:
        self._last = dict(steps)
        self.pose = Pose2D()
        self.path_length_m = 0.0

    def update(self, steps: Mapping[str, int]) -> tuple[float, float, float]:
        deltas = {name: unwrap_steps(self._last[name], steps[name]) for name in WHEEL_IDS}
        self._last = dict(steps)
        dx, dy, dtheta = wheel_steps_to_body(deltas)
        self.pose = integrate_pose(self.pose, dx, dy, dtheta)
        self.path_length_m += math.hypot(dx, dy)
        return dx, dy, dtheta


def _tick(clock: Clock, reader: _Reader, odometry: _Odometry, segment: Segment) -> float:
    """1 周期待って読み、積み上げる。この区間の向きへ進んだぶんを返す（読めなければ 0）。"""
    clock.sleep(PERIOD_S)
    steps = reader.read()
    if steps is None:
        return 0.0
    return segment.progress(*odometry.update(steps))


def _run_segment(
    index: int, segment: Segment, wheels: Wheels, clock: Clock, reader: _Reader, odometry: _Odometry
) -> tuple[SegmentResult, str | None]:
    error: str | None = None
    # 読みは PERIOD_S おきなので、目標に最も近い読みで止める。届いてから止めると、区間ごとに平均で
    # 半周期ぶん行き過ぎ、旋回の行き過ぎが積もって四角が閉じなくなる
    stop_at = segment.target - segment.rate * PERIOD_S / 2.0
    wheels.write_velocity(body_to_wheel_raw(segment.vx_mps, segment.vy_mps, segment.omega_dps))
    started = clock.now()
    progress = 0.0
    while progress < stop_at and not reader.gave_up:
        if clock.now() - started > segment.time_limit_s:
            error = f"{index} {segment.label}が時間の上限 {segment.time_limit_s:.1f} 秒を超えた"
            break
        progress += _tick(clock, reader, odometry, segment)
    wheels.stop()
    elapsed_s = clock.now() - started
    settle_until = clock.now() + SETTLE_S
    while clock.now() < settle_until and not reader.gave_up:
        progress += _tick(clock, reader, odometry, segment)
    if reader.gave_up:
        error = _READ_FAILED
    return SegmentResult(index, segment, progress, elapsed_s), error


def drive(
    wheels: Wheels,
    clock: Clock,
    plan: Sequence[Segment],
    on_segment: Callable[[SegmentResult], None] = lambda _: None,
) -> DriveResult:
    """計画の区間を順に走らせる。時間の上限か読みの失敗で止めたら、残りの区間へ進まずに返す。"""
    reader = _Reader(wheels)
    first = reader.read()
    while first is None and not reader.gave_up:
        clock.sleep(PERIOD_S)
        first = reader.read()
    if first is None:
        return DriveResult((), Pose2D(), 0.0, _READ_FAILED)
    odometry = _Odometry(first)
    results: list[SegmentResult] = []
    for index, segment in enumerate(plan, start=1):
        result, error = _run_segment(index, segment, wheels, clock, reader, odometry)
        results.append(result)
        on_segment(result)
        if error is not None:
            return DriveResult(tuple(results), odometry.pose, odometry.path_length_m, error)
    return DriveResult(tuple(results), odometry.pose, odometry.path_length_m)


def dry_run(
    plan: Sequence[Segment], on_segment: Callable[[SegmentResult], None] = lambda _: None
) -> DriveResult:
    """バスを開かず、指令どおりに回る擬似エンコーダで drive を回す。時計も擬似なので待たない。"""
    clock = SimClock()
    # 4095 と 0 の継ぎ目の手前から始め、巻き戻しを毎回通す
    wheels = SimWheels(clock, start=dict.fromkeys(WHEEL_IDS, STEPS_PER_TURN - 6))
    return drive(wheels, clock, plan, on_segment)


def open_bus(port: str) -> Any:
    """bus2 を開く。lerobot はここでだけ import する（Mac のテストと dry-run に lerobot は無い）。"""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for name, i in WHEEL_IDS.items()}
    bus = FeetechMotorsBus(port=port, motors=motors)
    # handshake は 3 台すべての応答を要求して例外になる。揃っているかは run_on_bus が確かめる
    bus.connect(handshake=False)
    return bus


def run_on_bus(
    bus: Any,
    plan: Sequence[Segment],
    clock: Clock,
    on_segment: Callable[[SegmentResult], None] = lambda _: None,
) -> DriveResult | None:
    """接続済みの bus2 で計画を走らせる。ID 7-9 が揃わなければ、何も書かずに閉じて None を返す。

    走らせたあとは、例外でも Ctrl+C でも、速度 0 を書いてから脱力する。
    """
    found = frozenset(bus.broadcast_ping() or {})
    if not set(WHEEL_IDS.values()) <= found:
        # bus1（左腕 ID1-6、head ID7-8）を指定すると、ID7-8 は head になる。
        # 閉じるときの脱力も head への書き込みなので、脱力せずに閉じる
        bus.disconnect(disable_torque=False)
        print(f"ID 7,8,9 が揃わない（応答: {sorted(found)}）。bus2 のポートか確認する")
        return None
    try:
        # Operating_Mode は EEPROM 側なので、Lock を落とさないと書き込めない。
        # disable_torque が Torque_Enable=0 と Lock=0 の両方を書く
        bus.disable_torque()
        for name in WHEEL_IDS:
            bus.write("Operating_Mode", name, 1, normalize=False)  # 1 = VELOCITY
        # 前の実行の Goal_Velocity が残っていると、トルクを入れた時点で回り出す
        bus.sync_write("Goal_Velocity", _zero_velocity(), num_retry=5)
        bus.enable_torque()
        return drive(_BusWheels(bus), clock, plan, on_segment)
    finally:
        # 回りっぱなしで抜けさせない。止めてから脱力する
        try:
            bus.sync_write("Goal_Velocity", _zero_velocity(), num_retry=5)
        finally:
            bus.disconnect(disable_torque=True)


def _format_raw(raw: Mapping[str, int]) -> str:
    return "  ".join(f"ID{WHEEL_IDS[name]} {value:+5d}" for name, value in raw.items())


def _format_amount(value: float, segment: Segment) -> str:
    return f"{value:.3f} m" if segment.kind == "straight" else f"{value:.1f} deg"


def _format_command(segment: Segment) -> str:
    if segment.kind == "turn":
        return f"theta {segment.omega_dps:+.1f} deg/s"
    command = f"x {segment.vx_mps:+.3f} m/s"
    if segment.vy_mps:
        command += f", y {segment.vy_mps:+.3f} m/s"
    return command


def format_plan(plan: Sequence[Segment]) -> str:
    lines = ["計画（カメラの側 = 前 = +x を先頭に進み、隅で左へ 90° 回る）"]
    for index, segment in enumerate(plan, start=1):
        raw = body_to_wheel_raw(segment.vx_mps, segment.vy_mps, segment.omega_dps)
        lines.append(
            f"  {index} {segment.label} {_format_amount(segment.target, segment):>9s}"
            f"  {_format_command(segment):18s}  raw {_format_raw(raw)}  上限 {segment.time_limit_s:.1f} 秒"
        )
    return "\n".join(lines)


def format_segment(result: SegmentResult) -> str:
    segment = result.segment
    return (
        f"  {result.index} {segment.label}  指令 {_format_amount(segment.target, segment)}"
        f"（{_format_command(segment)}）  走行 {_format_amount(result.progress, segment)}"
        f"  {result.elapsed_s:.1f} 秒"
    )


def format_result(result: DriveResult) -> str:
    distance_m, heading_deg = closure_error(result.pose)
    lines = [
        f"車輪オドメトリの経路長: {result.path_length_m:.3f} m",
        (
            f"車輪オドメトリの閉じ誤差: 始点から {distance_m:.3f} m、向きの差 {heading_deg:+.2f} deg"
            f"（終点 x {result.pose.x:+.3f} m、y {result.pose.y:+.3f} m）"
        ),
    ]
    if result.error is not None:
        lines.append(f"失敗: {result.error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", help="bus2（右腕 + ホイール）のポート。--dry-run では使わない")
    parser.add_argument("--distance", type=float, default=2.0, help="1 辺の長さ m（既定 2.0）")
    parser.add_argument("--sides", type=int, default=4, help="辺の数（既定 4）。各辺のあとに左へ 90° 回る")
    parser.add_argument(
        "--speed", type=float, default=0.1, help=f"直進速度 m/s（既定 0.1、上限 {MAX_SPEED_MPS:g}）"
    )
    parser.add_argument(
        "--turn-speed", type=float, default=20.0, help=f"旋回速度 deg/s（既定 20、上限 {MAX_TURN_DPS:g}）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="バスを開かず、擬似エンコーダで同じループを回す"
    )
    args = parser.parse_args(argv)
    if args.port is None and not args.dry_run:
        parser.error("--port か --dry-run のどちらかが要る")
    try:
        plan = plan_square(args.distance, args.sides, args.speed, args.turn_speed)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print("dry-run: バスを開かず、擬似エンコーダで回す")
    # SSH 越しでは標準出力がバッファされる。動き出す前に計画を見せる
    print(format_plan(plan), flush=True)

    def report(result: SegmentResult) -> None:
        print(format_segment(result), flush=True)

    if args.dry_run:
        result = dry_run(plan, report)
    else:
        try:
            bus = open_bus(args.port)
        except ModuleNotFoundError:
            print("lerobot が無い。Raspberry Pi の venv で走らせるか、--dry-run を付ける", file=sys.stderr)
            return 1
        except ConnectionError as exc:
            print(f"{args.port} を開けない: {exc}", file=sys.stderr)
            return 1
        try:
            result = run_on_bus(bus, plan, MonotonicClock(), report)
        except KeyboardInterrupt:
            print("Ctrl+C で止めた。停止・脱力した")
            return 1
        if result is None:
            return 1
        print("停止・脱力した")
    print(format_result(result))
    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
