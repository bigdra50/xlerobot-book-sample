"""Raspberry Pi で走らせる常駐プロセス。ホイール 3 台（bus2 の ID 7〜9）だけを受け持ち、
TCP で速度の指令を受けながら、エンコーダの値を流し続ける。

    ~/xlerobot-book-env/venv/bin/python base_server.py --port /dev/serial/by-id/<bus2> [--listen 0.0.0.0:8766]

1 行に 1 つの JSON でやり取りする。

- 受け取る: {"vx": m/s, "vy": m/s, "wz": deg/s}（車体座標。向きは公式 _body_to_wheel_raw と同じ）
- 送る: {"t": Pi のシステム時刻 [s], "pos": {...}, "vel": {...}, "cur": {...}}（キーは ID の文字列）と、{"event": ...}

安全の約束:

- ID 7〜9 が揃わなければ、何も書かずに終わる（bus1 を開いても head へ書かない）
- 指令が 0.3 秒届かなければ止める。接続が切れたら止めて脱力する。受ける接続は 1 つだけ
- 指令が変わってから 0.4 秒を除き、電流が 3 回続けて上限を超えるか、指令の 30 % に届かない輪が 5 回続いたら止める。
  止めたあとは、速度 0 の指令が来るまで動かさない
- 読みが 5 回続けて失敗したら、止めて終わる
- 本書の host（host/xlerobot_host.py）とは bus2 を取り合うので、同時に動かさない
"""

from __future__ import annotations

import argparse
import json
import math
import select
import signal
import socket
import time
from dataclasses import dataclass

# 式と定数を二重に持たないため、同じディレクトリの drive_base.py から import する（drive_square.py と同じ置き方）
from drive_base import WHEEL_IDS, body_to_wheel_raw

ZERO_CMD = (0.0, 0.0, 0.0)
MAX_READ_FAILURES = 5
SEND_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class Limits:
    max_linear_mps: float
    max_angular_dps: float


def clamp_command(msg: object, lim: Limits) -> tuple[float, float, float] | None:
    """指令を (vx, vy, wz) にし、上限で抑える。形が崩れた指令は None（止める）。

    並進は向きを保ったまま大きさだけを縮める。成分ごとに切ると、斜めの指令の向きが変わる。
    """
    if not isinstance(msg, dict):
        return None
    vals = []
    for key in ("vx", "vy", "wz"):
        v = msg.get(key, 0.0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None
        vals.append(float(v))
    vx, vy, wz = vals
    speed = math.hypot(vx, vy)
    if speed > lim.max_linear_mps:
        k = lim.max_linear_mps / speed
        vx, vy = vx * k, vy * k
    wz = max(-lim.max_angular_dps, min(lim.max_angular_dps, wz))
    return vx, vy, wz


class Watchdog:
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.last: float | None = None

    def feed(self, now: float) -> None:
        self.last = now

    def expired(self, now: float) -> bool:
        return self.last is None or now - self.last > self.timeout_s


class Guard:
    """何かに当たったことを、電流と、指令に追いつかない輪から見つける。"""

    def __init__(
        self, max_current: int, min_ratio: float, over_limit: int, slow_limit: int, grace_s: float
    ) -> None:
        self.max_current = max_current
        self.min_ratio = min_ratio
        self.over_limit = over_limit
        self.slow_limit = slow_limit
        self.grace_s = grace_s
        self.over = 0
        self.slow = 0

    def check(self, row: dict, targets: dict[str, int], since_change_s: float) -> str | None:
        if since_change_s < self.grace_s:
            # 動き出しは電流が上がり、速さも指令に届いていないのが普通なので数えない
            self.over = self.slow = 0
            return None
        self.over = self.over + 1 if max(row["cur"].values()) > self.max_current else 0
        if self.over >= self.over_limit:
            return f"current over {self.max_current}"
        lagging = [
            k for k, t in targets.items() if abs(t) > 100 and abs(row["vel"][k]) < self.min_ratio * abs(t)
        ]
        self.slow = self.slow + 1 if lagging else 0
        if self.slow >= self.slow_limit:
            return f"wheel {lagging} below {int(self.min_ratio * 100)}% of command"
        return None


def row_line(t: float, pos: dict, vel: dict, cur: dict) -> str:
    return json.dumps({"t": t, "pos": pos, "vel": vel, "cur": cur}) + "\n"


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


class ClientClosed(Exception):
    """相手が接続を閉じた。"""


class BaseSession:
    """1 つの接続の、指令の受け取り、見張り、読みの送り出し。

    バスには触れない。書く値（targets）を返し、読んだ値（report）を受け取るだけにしてある。
    base_server と本書の host（host/xlerobot_host.py）の両方が、同じやり取りをこれで持つ。
    """

    def __init__(self, conn: socket.socket, lim: Limits, max_current: int, rate_hz: float) -> None:
        self.conn = conn
        self.lim = lim
        self.rate_hz = rate_hz
        self.names = list(WHEEL_IDS)
        self.ids = {n: str(WHEEL_IDS[n]) for n in self.names}
        self.dog = Watchdog(0.3)
        self.guard = Guard(max_current, 0.3, 3, 5, 0.4)
        self.cmd = ZERO_CMD
        self.latched = False
        self.current = dict.fromkeys(self.names, 0)
        self.changed_at = 0.0
        self.buf = b""
        conn.settimeout(SEND_TIMEOUT_S)

    def send(self, obj: dict) -> None:
        self.conn.sendall((json.dumps(obj) + "\n").encode())

    def begin(self) -> None:
        self.send({"event": "ready", "rate_hz": self.rate_hz})

    def targets(self, now: float) -> dict[str, int]:
        """届いた指令をすべて読み、いま書く車輪の値を返す。相手が閉じていたら ClientClosed。"""
        while select.select([self.conn], [], [], 0)[0]:
            chunk = self.conn.recv(4096)
            if not chunk:
                raise ClientClosed
            self.buf += chunk
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                try:
                    got = clamp_command(json.loads(line), self.lim)
                except json.JSONDecodeError:
                    got = None
                self.cmd = ZERO_CMD if got is None else got
                self.dog.feed(now)
                if self.latched and self.cmd == ZERO_CMD:
                    self.latched = False
        want = ZERO_CMD if self.latched or self.dog.expired(now) else self.cmd
        new = body_to_wheel_raw(*want)
        if new != self.current:
            self.current = new
            self.changed_at = now
        return dict(self.current)

    def report(self, t_read: float, pos: dict, vel: dict, cur: dict, now: float) -> bool:
        """読んだ値を送る。当たったと判断したら止めた状態にして True を返す（呼ぶ側が速度 0 を書く）。"""
        row = {
            "pos": {self.ids[n]: int(pos[n]) for n in self.names},
            "vel": {self.ids[n]: int(vel[n]) for n in self.names},
            "cur": {self.ids[n]: int(cur[n]) for n in self.names},
        }
        tripped = False
        if not self.latched:
            why = self.guard.check(
                row, {self.ids[n]: v for n, v in self.current.items()}, now - self.changed_at
            )
            if why:
                self.latched = True
                self.current = dict.fromkeys(self.names, 0)
                tripped = True
                self.send({"event": "guard", "reason": why})
        self.conn.sendall(row_line(t_read, row["pos"], row["vel"], row["cur"]).encode())
        return tripped


def serve_client(bus, conn: socket.socket, args: argparse.Namespace) -> str:
    """1 つの接続のあいだ、指令を受けて車輪を回し、読みを送る。終わった理由を返す。"""
    names = list(WHEEL_IDS)
    zero = dict.fromkeys(names, 0)
    session = BaseSession(conn, Limits(args.max_linear, args.max_angular), args.max_current, args.rate)
    failures = 0
    period = 1.0 / args.rate
    next_t = time.monotonic()
    bus.sync_write("Goal_Velocity", zero, num_retry=5)
    bus.enable_torque(num_retry=3)
    session.begin()
    while True:
        now = time.monotonic()
        try:
            targets = session.targets(now)
        except ClientClosed:
            return "client closed"
        bus.sync_write("Goal_Velocity", targets)
        t_read = time.time()
        try:
            pos = bus.sync_read("Present_Position", names, normalize=False, num_retry=1)
            vel = bus.sync_read("Present_Velocity", names, normalize=False, num_retry=1)
            cur = bus.sync_read("Present_Current", names, normalize=False, num_retry=1)
            failures = 0
        except ConnectionError as exc:
            failures += 1
            if failures >= MAX_READ_FAILURES:
                session.send({"event": "error", "reason": f"read failures: {exc}"[:120]})
                return "read failures"
        else:
            if session.report(t_read, pos, vel, cur, now):
                bus.sync_write("Goal_Velocity", zero, num_retry=5)
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="bus2 のシリアルポート（by-id のパスを渡す）")
    ap.add_argument("--listen", default="0.0.0.0:8766")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--max-linear", type=float, default=0.2, help="m/s")
    ap.add_argument("--max-angular", type=float, default=45.0, help="deg/s")
    ap.add_argument("--max-current", type=int, default=120, help="Present_Current の生の値（1 = 6.5 mA）")
    args = ap.parse_args()

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, _interrupt)
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in WHEEL_IDS.items()}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect(handshake=False)
    found = frozenset(bus.broadcast_ping() or {})
    if not set(WHEEL_IDS.values()) <= found:
        bus.disconnect(disable_torque=False)
        print(
            json.dumps({"event": "error", "reason": "wheel ids missing", "found": sorted(found)}), flush=True
        )
        return 1
    host, port = args.listen.rsplit(":", 1)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(1)
    print(json.dumps({"event": "listening", "listen": args.listen}), flush=True)
    names = list(WHEEL_IDS)
    zero = dict.fromkeys(names, 0)
    try:
        bus.disable_torque(num_retry=3)
        for n in names:
            # Operating_Mode は EEPROM。すでに速度モードなら書かない（フラッシュの書き換えを増やさない）
            if bus.read("Operating_Mode", n, normalize=False, num_retry=3) != 1:
                bus.write("Operating_Mode", n, 1, normalize=False, num_retry=3)
        while True:
            conn, peer = srv.accept()
            print(json.dumps({"event": "connected", "peer": peer[0]}), flush=True)
            try:
                reason = serve_client(bus, conn, args)
            except OSError as exc:
                reason = f"connection error: {exc}"
            finally:
                bus.sync_write("Goal_Velocity", zero, num_retry=5)
                bus.disable_torque(num_retry=3)
                conn.close()
            print(json.dumps({"event": "disconnected", "reason": reason}), flush=True)
            if reason == "read failures":
                return 1
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            bus.sync_write("Goal_Velocity", zero, num_retry=5)
        finally:
            bus.disconnect(disable_torque=True)
            srv.close()
            print(json.dumps({"event": "released"}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
