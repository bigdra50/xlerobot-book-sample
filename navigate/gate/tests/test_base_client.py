import json
import socket
import threading
import time

from navgate.base_client import BaseClient


class FakeServer:
    """1 接続だけ受け、送られてきた行を貯め、決めた行を送り返す。"""

    def __init__(self, lines_to_send: list[str]) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.received: list[dict] = []
        self.lines_to_send = lines_to_send
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        conn, _ = self.sock.accept()
        with conn:
            for line in self.lines_to_send:
                conn.sendall(line.encode())
            conn.settimeout(1.0)
            buf = b""
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        one, buf = buf.split(b"\n", 1)
                        self.received.append(json.loads(one))
            except TimeoutError:
                pass


def test_rows_and_events_arrive_with_arrival_times():
    server = FakeServer(
        ['{"t": 1.0, "pos": {"7": 1, "8": 2, "9": 3}}\n', '{"event": "guard", "reason": "current"}\n']
    )
    rows, events = [], []
    client = BaseClient("127.0.0.1", server.port, on_row=rows.append, on_event=events.append)
    deadline = time.monotonic() + 2.0
    while (not rows or not events) and time.monotonic() < deadline:
        time.sleep(0.01)
    client.close()
    assert rows[0]["pos"] == {"7": 1, "8": 2, "9": 3}
    assert "t_arrival" in rows[0]
    assert events[0]["reason"] == "current"


def test_velocity_commands_are_sent_as_json_lines():
    server = FakeServer([])
    client = BaseClient("127.0.0.1", server.port)
    client.send_velocity(0.1, -0.05, 20.0)
    client.stop()
    time.sleep(0.2)
    client.close()
    server.thread.join(timeout=2.0)
    assert server.received[0] == {"vx": 0.1, "vy": -0.05, "wz": 20.0}
    assert server.received[1] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
