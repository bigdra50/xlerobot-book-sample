#!/usr/bin/env python
# 元: Vector-Wangel/XLeRobot software/plugins/lerobot_robot_xlerobot/lerobot_robot_xlerobot/xlerobot_host.py（Apache-2.0）
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# 変更点は下の docstring にある
"""公式の host（ホスト側の ZMQ サーバ）を、引数で設定して起動する。

公式 `xlerobot_host.py` の main() は
`XLerobotConfig(id="my_xlerobot_pc")` を固定で作る。
そのため id を変えられず、カメラは既定の空のまま、単位も use_degrees=False 固定になる。
client 側で `--robot.cameras` を指定しても、host が送らなければ映像は届かない。

ループ本体は公式（Vector-Wangel/XLeRobot、Apache-2.0）をもとに、次を足した。

- トルクを入れる前に、腕と head の目標の位置を今の位置へそろえる（起動した瞬間に腕が跳ばない）
- 観測の読み取りが失敗しても、続けて MAX_OBSERVATION_FAILURES 回まではその周を飛ばして動き続ける
- --rest.path を渡すと、止めるときに腕と head を rest の姿勢へ戻してから脱力する（公式はその場で脱力し、腕が自重で落ちる）。
  --rest.capture=true なら、手で置いた今の姿勢を --rest.path へ保存して終わる
- --base.listen を渡すと、車輪の口を開く（navigate/gate/base_server.py と同じやり取り）。
  つながっているあいだは台車をこの口の指令で動かし、車輪のエンコーダを流す。
  ZMQ の client の台車の指令は無視し、腕と head の指令はそのまま通す
- --observe.port を渡すと、同じ観測を PUB でも配る。
  公式の観測の口（PUSH）は、つないだ相手が複数いると観測を順番に振り分けるので、
  見るだけの相手は、こちらへつなぐ
- 観測の口の OS の送信バッファを絞る（--observe.sndbuf_kb）。
  公式は OS の既定のままで、Wi-Fi で送りきれない観測が送信バッファにたまり、どの観測も遅れて届く

--base.listen と --observe.port は付録の navigate/ のための口で、VR テレオペと記録には要らない。

    python xlerobot_host.py --robot.id=follower \
        --robot.port1=/dev/ttyACM0 --robot.port2=/dev/ttyACM1 \
        --robot.cameras='{head: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
        --base.listen=0.0.0.0:8766 --observe.port=5557

ホスト（Raspberry Pi など）で、install.sh が作った venv から実行する。
"""

# draccus は関数注釈から config クラスを取るので、from __future__ import annotations は使えない
import base64
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import zmq
from host_base import BasePort, capture_rest, hold_goal_at_present, load_rest, move_to_rest
from lerobot_robot_xlerobot import XLerobot
from lerobot_robot_xlerobot.config_xlerobot import XLerobotConfig, XLerobotHostConfig
from lerobot_robot_xlerobot.xlerobot_host import XLerobotHost

# 公式のループは、バスかカメラの読み取りが 1 回失敗しただけで例外で終わる。
# 単発の失敗（応答の取りこぼし、カメラの 1 枚の遅れ）では止めず、続けて失敗したときだけ止める。30 Hz で約 0.2 秒
MAX_OBSERVATION_FAILURES = 5

logger = logging.getLogger(__name__)


@dataclass
class BaseCLI:
    # 空なら車輪の口を開かない（公式と同じ動き）
    listen: str = ""
    max_linear: float = 0.2  # m/s
    max_angular: float = 45.0  # deg/s
    max_current: int = 120  # Present_Current の生の値（1 = 6.5 mA）


@dataclass
class ObserveCLI:
    # 0 なら開かない
    port: int = 0
    # 観測の口（PUSH）の OS の送信バッファ（KB）。0 なら OS に任せる。
    # 観測 1 件はカメラ 3 台で約 240 KB。Linux は指定の 2 倍を確保するので、256 なら約 2 件までしかためない。
    # OS に任せると、Wi-Fi で送りきれない分が 1.5〜1.8 MB たまり、観測が約 0.3 秒遅れて届いた
    sndbuf_kb: int = 256


@dataclass
class RestCLI:
    # 止めるときに戻る姿勢の JSON（モーター名 → キャリブレーションの範囲の中央を 0 とした度）。空なら戻さない
    path: str = ""
    # true なら、手で置いた今の姿勢を path へ保存して終わる
    capture: bool = False


class BookXLerobot(XLerobot):
    # 車輪の口がつながっているあいだの車輪の値。None なら公式どおり action の x.vel などから決める
    wheel_override: dict[str, int] | None = None

    def _body_to_wheel_raw(self, *args, **kwargs):
        # 公式の send_action は、action に台車の速度が無いと 0 として車輪へ書く。
        # 車輪の口が台車を受け持つあいだは、その口の値で置き換える（腕の指令だけの client でも台車が止まらない）
        if self.wheel_override is not None:
            return dict(self.wheel_override)
        return super()._body_to_wheel_raw(*args, **kwargs)

    def configure(self):
        hold_goal_at_present(self.bus1, self.left_arm_motors + self.head_motors)
        hold_goal_at_present(self.bus2, self.right_arm_motors)
        super().configure()


class BookXLerobotHost(XLerobotHost):
    """公式の XLerobotHost と同じ口を開く。観測の口だけ、bind の前に OS の送信バッファを絞る。"""

    def __init__(self, config: XLerobotHostConfig, sndbuf_kb: int = 0):
        # 送信バッファは bind の前に決める必要があるので、公式の __init__ を呼ばずに同じ手順で作る
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        if sndbuf_kb > 0:
            self.zmq_observation_socket.setsockopt(zmq.SNDBUF, sndbuf_kb * 1024)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz


@dataclass
class HostCLI:
    robot: XLerobotConfig = field(default_factory=lambda: XLerobotConfig(id="my_xlerobot_pc"))
    # 公式の既定 connection_time_s=3600 は、起動から 1 時間でループを抜けて脱力する。
    # 収集セッションの途中で止まるので、既定を 24 時間にする
    host: XLerobotHostConfig = field(default_factory=lambda: XLerobotHostConfig(connection_time_s=24 * 3600))
    base: BaseCLI = field(default_factory=BaseCLI)
    observe: ObserveCLI = field(default_factory=ObserveCLI)
    rest: RestCLI = field(default_factory=RestCLI)


def serve(
    robot: BookXLerobot, host: XLerobotHost, port: BasePort | None = None, pub: zmq.Socket | None = None
) -> None:
    last_cmd_time = time.time()
    watchdog_active = False
    start = time.perf_counter()
    duration = 0.0
    read_failures = 0
    logger.info("Waiting for commands...")
    while duration < host.connection_time_s:
        loop_start_time = time.time()
        now_m = time.monotonic()
        wheels = port.poll(now_m, bus=robot.bus2) if port is not None else None
        robot.wheel_override = wheels
        wheels_written = False
        try:
            msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
            robot.send_action(dict(json.loads(msg)))
            wheels_written = wheels is not None
            last_cmd_time = time.time()
            watchdog_active = False
        except zmq.Again:
            pass
        except Exception as e:  # noqa: BLE001
            logger.error("Message fetching failed: %s", e)

        now = time.time()
        if wheels is not None:
            # 台車は車輪の口が受け持つ。その口にも 0.3 秒の見張りがある。
            # client の指令が届いた周は、send_action が wheel_override の値を書いているので書き直さない
            if not wheels_written:
                robot.bus2.sync_write("Goal_Velocity", wheels)
        elif (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
            logger.warning("Command not received for %d ms. Stopping the base.", host.watchdog_timeout_ms)
            watchdog_active = True
            robot.stop_base()

        try:
            observation = robot.get_observation()
        except (ConnectionError, TimeoutError) as e:
            read_failures += 1
            if read_failures > MAX_OBSERVATION_FAILURES:
                raise
            logger.warning("Observation read failed (%d/%d): %s", read_failures, MAX_OBSERVATION_FAILURES, e)
            time.sleep(1 / host.max_loop_freq_hz)
            duration = time.perf_counter() - start
            continue
        read_failures = 0
        if port is not None:
            port.report(robot.bus2, robot.base_motors, now_m)
            for line in port.log:
                logger.info(line)
            port.log.clear()
        for cam_key in robot.cameras:
            ok, buffer = cv2.imencode(".jpg", observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            observation[cam_key] = base64.b64encode(buffer).decode("utf-8") if ok else ""
        message = json.dumps(observation)
        try:
            host.zmq_observation_socket.send_string(message, flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.info("Dropping observation, no client connected")
        if pub is not None:
            pub.send_string(message, flags=zmq.NOBLOCK)

        elapsed = time.time() - loop_start_time
        time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
        duration = time.perf_counter() - start


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


@draccus.wrap()
def main(cfg: HostCLI) -> None:
    logging.basicConfig(level=logging.INFO)
    # kill（SIGTERM）でも finally の脱力を通す。ハンドラが無いとトルク保持のまま残る
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    if cfg.rest.capture and not cfg.rest.path:
        raise SystemExit("--rest.capture=true には --rest.path が要る")
    robot = BookXLerobot(cfg.robot)
    robot.connect()
    if cfg.rest.capture:
        rest = capture_rest(robot)
        Path(cfg.rest.path).write_text(json.dumps(rest, indent=2) + "\n")
        logger.info("rest の姿勢を保存した: %s", cfg.rest.path)
        robot.disconnect()
        return
    host = BookXLerobotHost(cfg.host, sndbuf_kb=cfg.observe.sndbuf_kb)
    port = None
    if cfg.base.listen:
        port = BasePort(
            cfg.base.listen,
            cfg.base.max_linear,
            cfg.base.max_angular,
            cfg.base.max_current,
            float(cfg.host.max_loop_freq_hz),
        )
        logger.info("Wheel port listening on %s", cfg.base.listen)
    pub = None
    if cfg.observe.port:
        pub = host.zmq_context.socket(zmq.PUB)
        # 遅れた相手には最新の観測だけを渡す
        pub.setsockopt(zmq.SNDHWM, 1)
        pub.setsockopt(zmq.CONFLATE, 1)
        pub.bind(f"tcp://*:{cfg.observe.port}")
    try:
        serve(robot, host, port, pub)
    except KeyboardInterrupt:
        pass
    finally:
        if port is not None:
            port.close(robot.bus2)
        if pub is not None:
            pub.close(linger=0)
        if cfg.rest.path:
            try:
                move_to_rest(robot, load_rest(Path(cfg.rest.path)), float(cfg.host.max_loop_freq_hz))
            except Exception as e:  # noqa: BLE001  rest へ戻れなくても脱力と後始末は続ける
                logger.warning("rest の姿勢へ戻れなかった: %s", e)
        robot.disconnect()
        host.disconnect()


if __name__ == "__main__":
    main()
