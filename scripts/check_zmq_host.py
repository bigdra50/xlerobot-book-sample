"""ホストに載せた公式 host が観測を流しているかを、client を起動する前に確かめる。

公式 client の connect() は観測が 5 秒来なければ例外を投げるだけで、理由は分からない。

    DeviceNotConnectedError: Timeout waiting for LeKiwi Host to connect expired.

この 1 行は「host が落ちている」「経路が通っていない」「ポートが塞がっている」の
どれでも同じ文面で出る。観測ソケット (5556) だけを先に覗いて、どこで切れているかを
決めるためのスクリプト。モーターには一切触らない。

判定は 3 つ。

  受信が 0 件            -> host かポートか経路。nc -z <host> 5556 で到達を分ける
  受信はあるが次元が違う -> host 側のバス構成（17 台そろっていない）
  受信も次元も正常       -> client を起動してよい

Usage:
    python check_zmq_host.py --host xlerobot-pi.local
    python check_zmq_host.py --host 192.168.0.20 --port 5556 --seconds 3

依存は pyzmq のみ（lerobot は import しない）。
Exit code: PASS なら 0、受信なしか次元不一致なら 1。
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Final

import zmq

# 腕 12 + head 2 = 14 の .pos と、車体速度 3 の .vel
EXPECTED_POS_KEYS: Final[int] = 14
EXPECTED_VEL_KEYS: Final[int] = 3
# 公式 host の観測ソケットは PUSH。既定ポートは 5556
DEFAULT_PORT: Final[int] = 5556


def collect(host: str, port: int, seconds: float) -> list[dict[str, Any]]:
    """観測を seconds 秒ぶん受け取る。CONFLATE は付けない（件数を数えたいため）。"""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://{host}:{port}")
    received: list[dict[str, Any]] = []
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            if not dict(poller.poll(200)):
                continue
            try:
                received.append(dict(json.loads(sock.recv_string(zmq.NOBLOCK))))
            except zmq.Again:
                continue
    finally:
        sock.close()
        ctx.term()
    return received


def summarize(obs: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """観測 1 件を、関節・車体速度・カメラの 3 つに仕分ける。"""
    pos = sorted(k for k in obs if k.endswith(".pos"))
    vel = sorted(k for k in obs if k.endswith(".vel"))
    # カメラは base64 の JPEG 文字列で入る。数値キー以外を残す
    cams = sorted(k for k in obs if k not in pos and k not in vel)
    return pos, vel, cams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="ホスト（Raspberry Pi）のホスト名か IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="観測ソケット（既定 5556）")
    parser.add_argument("--seconds", type=float, default=5.0, help="受信する秒数")
    args = parser.parse_args()

    received = collect(args.host, args.port, args.seconds)
    if not received:
        print(f"observation 受信: 0 件（{args.seconds:.0f} 秒）")
        print(f"host が動いているか、nc -z {args.host} {args.port} で到達を確かめる")
        return 1

    hz = len(received) / args.seconds
    pos, vel, cams = summarize(received[-1])
    print(f"observation 受信: {hz:.1f} Hz（{len(received)} 件）")
    print(f"  関節キー {len(pos)} 個 + base {len(vel)} 個 = {len(pos) + len(vel)}")
    print(f"  カメラ {len(cams)} 台" + (f"（{', '.join(cams)}）" if cams else ""))

    if len(pos) != EXPECTED_POS_KEYS or len(vel) != EXPECTED_VEL_KEYS:
        print(f"\n次元が合わない（期待 {EXPECTED_POS_KEYS} + {EXPECTED_VEL_KEYS}）")
        print("host 側で 17 台そろっているかを check_bus.py で確かめる")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
