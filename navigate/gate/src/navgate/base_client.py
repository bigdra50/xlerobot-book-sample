"""Pi の base_server.py へつなぐクライアント。速度の指令を送り、エンコーダの行とイベントを受け取る。"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable


class BaseClient:
    def __init__(
        self,
        host: str,
        port: int,
        on_row: Callable[[dict], None] | None = None,
        on_event: Callable[[dict], None] | None = None,
        connect_timeout: float = 5.0,
    ) -> None:
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        self.sock.settimeout(0.5)
        self.on_row = on_row
        self.on_event = on_event
        self.closed = threading.Event()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = self.sock.recv(65536)
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    t_arrival = time.monotonic()
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    obj["t_arrival"] = t_arrival
                    callback = self.on_event if "event" in obj else self.on_row
                    if callback is not None:
                        callback(obj)
        finally:
            self.closed.set()

    def send_velocity(self, vx: float, vy: float, wz_dps: float) -> None:
        """車体座標の速度（m/s、m/s、deg/s）を送る。サーバーは 0.3 秒届かないと止めるので、動かすあいだは送り続ける。"""
        self._send({"vx": float(vx), "vy": float(vy), "wz": float(wz_dps)})

    def stop(self) -> None:
        self.send_velocity(0.0, 0.0, 0.0)

    def _send(self, obj: dict) -> None:
        with self._send_lock:
            self.sock.sendall((json.dumps(obj) + "\n").encode())

    def close(self) -> None:
        self._stop.set()
        try:
            self.stop()
        except OSError:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.thread.join(timeout=2.0)
