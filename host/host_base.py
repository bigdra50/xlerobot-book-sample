"""公式 host（xlerobot_host.py）に足す処理。バスとソケットの扱いを、host のループから分けてある。

- hold_goal_at_present: トルクを入れる前に、位置制御のモーターの目標を今の位置へそろえる
- move_to_rest、capture_rest: 止める前に腕と head を rest の姿勢へ戻す。その姿勢を保存する
- BasePort: navigate/gate/base_server.py と同じ口（TCP、1 行 1 JSON）で、車輪の指令を受けてエンコーダを流す
"""

from __future__ import annotations

import json
import logging
import math
import socket
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_WHEELS = ["base_left_wheel", "base_back_wheel", "base_right_wheel"]

# 手先を畳んでから根元を下ろす。伸びた腕のまま肩を動かすと重力の負荷が最大になり、肩のサーボが過負荷保護で止まる
REST_STAGES: tuple[tuple[str, ...], ...] = (
    ("elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
    ("shoulder_lift",),
    ("shoulder_pan",),
    ("head_motor_1", "head_motor_2"),
)
# 1 周あたりの目標の変化（raw）。30 Hz で約 40 度/秒
REST_STEP_RAW = 15
# 重力で残る偏差（約 30 raw）は着いたとみなす
REST_TOLERANCE_RAW = 40
# サーボの過電流保護は、上限を超えた電流が 2 秒（Over_Current_Protection_Time=200、10 ms 単位）続くと働く。
# 届かない段は、それより前に保持へ切り替える
REST_STAGE_TIMEOUT_S = 1.5
# キャリブレーションの範囲の端は、手で動かして止まった位置。目標は端から 50 raw（約 4 度）内側に置く
REST_RANGE_MARGIN_RAW = 50
# STS3215 の分解能 - 1。lerobot の度への換算と同じ
STEPS_PER_TURN = 4095


def hold_goal_at_present(bus, motors: list[str]) -> None:
    """目標の位置を今の位置に書く。

    電源を入れたあとのサーボは、目標の位置が 0 のままのことがある。
    そのままトルクを入れると、腕が 0 の位置へ跳ぶ。
    """
    if not motors:
        return
    present = bus.sync_read("Present_Position", motors, normalize=False, num_retry=3)
    bus.sync_write("Goal_Position", present, normalize=False, num_retry=3)


def deg_to_raw(deg: float, cal) -> int:
    """キャリブレーションの範囲の中央を 0 度とした角度を、raw の位置にする。範囲の外は端に丸める。"""
    raw = round((cal.range_min + cal.range_max) / 2 + deg * STEPS_PER_TURN / 360)
    return max(cal.range_min, min(cal.range_max, raw))


def raw_to_deg(raw: int, cal) -> float:
    return (raw - (cal.range_min + cal.range_max) / 2) * 360 / STEPS_PER_TURN


def _inside_range(raw: int, cal) -> int:
    return max(cal.range_min + REST_RANGE_MARGIN_RAW, min(cal.range_max - REST_RANGE_MARGIN_RAW, raw))


def load_rest(path: Path) -> dict[str, float]:
    """rest の姿勢の JSON（モーター名 → 度）を読む。"""
    return {name: float(deg) for name, deg in json.loads(path.read_text()).items()}


def capture_rest(robot) -> dict[str, float]:
    """腕と head の今の位置を、rest の姿勢（モーター名 → 度）として返す。"""
    out: dict[str, float] = {}
    for bus, motors in (
        (robot.bus1, robot.left_arm_motors + robot.head_motors),
        (robot.bus2, robot.right_arm_motors),
    ):
        raw = bus.sync_read("Present_Position", motors, normalize=False, num_retry=3)
        out.update({m: round(raw_to_deg(v, robot.calibration[m]), 2) for m, v in raw.items()})
    return out


def move_to_rest(robot, rest_deg: dict[str, float], rate_hz: float) -> None:
    """腕と head を、REST_STAGES の段ごとに rest の姿勢へ動かす。脱力の直前に呼ぶ。"""
    for stage in REST_STAGES:
        moves = []
        for bus in (robot.bus1, robot.bus2):
            motors = [
                m for m in bus.motors if m in rest_deg and any(m == j or m.endswith("_" + j) for j in stage)
            ]
            if motors:
                start = bus.sync_read("Present_Position", motors, normalize=False, num_retry=3)
                goal = {
                    m: _inside_range(deg_to_raw(rest_deg[m], robot.calibration[m]), robot.calibration[m])
                    for m in motors
                }
                moves.append((bus, start, goal))
        if not moves:
            continue
        # 目標を少しずつ動かす。一度に書くとサーボの全速で振れる
        steps = max(1, math.ceil(max(abs(g[m] - s[m]) for _, s, g in moves for m in g) / REST_STEP_RAW))
        for i in range(1, steps + 1):
            for bus, start, goal in moves:
                step = {m: round(start[m] + (goal[m] - start[m]) * i / steps) for m in goal}
                bus.sync_write("Goal_Position", step, normalize=False)
            time.sleep(1 / rate_hz)
        deadline = time.monotonic() + REST_STAGE_TIMEOUT_S
        while time.monotonic() < deadline:
            reached = True
            for bus, _, goal in moves:
                present = bus.sync_read("Present_Position", list(goal), normalize=False, num_retry=3)
                reached &= all(abs(present[m] - goal[m]) <= REST_TOLERANCE_RAW for m in goal)
            if reached:
                break
            time.sleep(0.05)
        else:
            # 機構に当たって届かない目標を押し続けると、サーボが過電流で止まる。届いた位置で保持へ切り替える
            for bus, _, goal in moves:
                present = bus.sync_read("Present_Position", list(goal), normalize=False, num_retry=3)
                bus.sync_write("Goal_Position", present, normalize=False)
                logger.warning("rest に届かなかったので今の位置で止めた: %s", sorted(goal))


def _base_server() -> ModuleType:
    """navigate/gate/base_server.py を読む。

    車輪の口のやり取りは、そこにある BaseSession をそのまま使う（ROS 2 の base_bridge がそのままつながる）。
    --base.listen を使うときだけ読むので、付録の navigate/ が無くても host は動く。
    """
    sys.path.append(str(Path(__file__).resolve().parents[1] / "navigate" / "gate"))
    import base_server

    return base_server


class BasePort:
    """車輪の口。接続は 1 つだけ受け、つながっているあいだは台車をこの口の指令で動かす。"""

    def __init__(
        self, listen: str, max_linear: float, max_angular: float, max_current: int, rate_hz: float
    ) -> None:
        self._bs = _base_server()
        host, port = listen.rsplit(":", 1)
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, int(port)))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.bound_port = self.srv.getsockname()[1]
        self.lim = self._bs.Limits(max_linear, max_angular)
        self.max_current = max_current
        self.rate_hz = rate_hz
        self.session: Any = None  # base_server.BaseSession。つながっていなければ None
        self.peer = ""
        self.failures = 0
        self.log: list[str] = []

    def _accept(self) -> None:
        try:
            conn, peer = self.srv.accept()
        except BlockingIOError:
            return
        conn.setblocking(True)
        self.session = self._bs.BaseSession(conn, self.lim, self.max_current, self.rate_hz)
        self.peer = peer[0]
        self.failures = 0
        self.session.begin()
        self.log.append(f"base client connected: {self.peer}")

    def _end(self, bus, reason: str, wheels: list[str] | None = None) -> None:
        if self.session is None:
            return
        try:
            self.session.conn.close()
        except OSError:
            pass
        self.session = None
        if bus is not None:
            bus.sync_write("Goal_Velocity", dict.fromkeys(wheels or _WHEELS, 0), num_retry=5)
        self.log.append(f"base client left: {reason}")

    def poll(self, now: float, bus=None) -> dict[str, int] | None:
        """車輪に書く値を返す。つながっている相手がいなければ None（台車は host の既定の扱いに戻る）。"""
        if self.session is None:
            self._accept()
            if self.session is None:
                return None
        try:
            return self.session.targets(now)
        except (self._bs.ClientClosed, OSError) as exc:
            self._end(bus, f"{type(exc).__name__}")
            return None

    def report(self, bus, wheels: list[str], now: float) -> None:
        """車輪の位置、速度、電流を読んで送る。当たったと判断されたら速度 0 を書く。"""
        if self.session is None:
            return
        t_read = time.time()
        try:
            pos = bus.sync_read("Present_Position", wheels, normalize=False, num_retry=1)
            vel = bus.sync_read("Present_Velocity", wheels, normalize=False, num_retry=1)
            cur = bus.sync_read("Present_Current", wheels, normalize=False, num_retry=1)
        except ConnectionError as exc:
            self.failures += 1
            if self.failures >= self._bs.MAX_READ_FAILURES:
                try:
                    self.session.send({"event": "error", "reason": f"read failures: {exc}"[:120]})
                except OSError:
                    pass
                self._end(bus, "read failures", wheels)
            return
        self.failures = 0
        try:
            if self.session.report(t_read, pos, vel, cur, now):
                bus.sync_write("Goal_Velocity", dict.fromkeys(wheels, 0), num_retry=5)
        except OSError as exc:
            self._end(bus, f"send failed: {exc}", wheels)

    def close(self, bus) -> None:
        self._end(bus, "host stopped")
        self.srv.close()
