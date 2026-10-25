"""Pi の base_server.py とつなぎ、/cmd_vel で車輪を回し、エンコーダの読みから /wheel/odom を出す。

base_server は指令が 0.3 秒届かないと止まるので、/cmd_vel の最後の値を 20 Hz で送り続ける。
/cmd_vel が cmd_timeout 秒途切れたら、速度 0 を送る。
/wheel/odom の時刻は、Pi がエンコーダを読んだ時刻を、このマシンの時計へ直したもの。
Pi と、Docker Desktop の中の時計は、NTP で合わせていても 50 ms ほどずれていた。
直近の行の「届いた時刻 - Pi の時刻」の最小値を、時計のずれとみなして足す。
ネットワークの最小の遅れの分だけ、時刻は遅めに出る。
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from navgate.base_client import BaseClient
from navgate.wheel_odom import WheelOdometry
from rclpy.node import Node

from xlerobot_nav.msgs import to_stamp, yaw_to_quaternion


class BaseBridge(Node):
    def __init__(self) -> None:
        super().__init__("base_bridge")
        host = str(self.declare_parameter("host", "xlerobot-pi.local").value)
        port = int(self.declare_parameter("port", 8766).value)
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.5).value)
        self.odom_frame = str(self.declare_parameter("odom_frame", "odom").value)
        self.base_frame = str(self.declare_parameter("base_frame", "base_link").value)
        self.odo = WheelOdometry(yaw_scale=float(self.declare_parameter("yaw_scale", 1.0).value))
        self.pub = self.create_publisher(Odometry, "wheel/odom", 50)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_t: float | None = None
        self._rows: deque[dict] = deque(maxlen=1000)
        self._prev_t: float | None = None
        self._delay: deque[float] = deque(maxlen=500)  # 届いた時刻 - Pi の時刻。直近 10 秒ぶん
        self.client = BaseClient(host, port, on_row=self._rows.append, on_event=self._on_event)
        self.get_logger().info(f"connected to base_server {host}:{port}")
        self.create_timer(0.05, self._send_tick)
        self.create_timer(0.01, self._drain)
        self.create_timer(10.0, self._report)

    def _on_cmd(self, msg: Twist) -> None:
        with self._lock:
            self._cmd = (msg.linear.x, msg.linear.y, math.degrees(msg.angular.z))
            self._cmd_t = time.monotonic()

    def _send_tick(self) -> None:
        with self._lock:
            fresh = self._cmd_t is not None and time.monotonic() - self._cmd_t < self.cmd_timeout
            cmd = self._cmd if fresh else (0.0, 0.0, 0.0)
        try:
            self.client.send_velocity(*cmd)
        except OSError as exc:
            self.get_logger().error(f"base_server is gone: {exc}")
            rclpy.try_shutdown()

    def _on_event(self, event: dict) -> None:
        self.get_logger().warning(f"base_server: {event}")

    def _drain(self) -> None:
        while self._rows:
            row = self._rows.popleft()
            self._delay.append(time.time() - row["t"])
            delta = self.odo.update(row["pos"])
            t = float(row["t"]) + min(self._delay)
            prev_t, self._prev_t = self._prev_t, t
            if delta is None or prev_t is None:
                continue
            dt = max(1e-3, t - prev_t)
            msg = Odometry()
            msg.header.stamp = to_stamp(t)
            msg.header.frame_id = self.odom_frame
            msg.child_frame_id = self.base_frame
            pose = self.odo.pose
            msg.pose.pose.position.x = pose.x
            msg.pose.pose.position.y = pose.y
            msg.pose.pose.orientation = yaw_to_quaternion(pose.theta)
            msg.twist.twist.linear.x = delta[0] / dt
            msg.twist.twist.linear.y = delta[1] / dt
            msg.twist.twist.angular.z = delta[2] / dt
            self.pub.publish(msg)

    def _report(self) -> None:
        if self._delay:
            d = sorted(self._delay)
            self.get_logger().info(
                f"row arrival - Pi time: min {d[0] * 1e3:.1f} ms (used as the clock offset), "
                f"median {statistics.median(d) * 1e3:.1f} ms"
            )

    def close(self) -> None:
        try:
            self.client.stop()
        except OSError:
            pass
        self.client.close()


def main() -> None:
    rclpy.init()
    node = BaseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
