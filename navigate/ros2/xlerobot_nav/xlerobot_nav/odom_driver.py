"""融合した /odom を見ながら /cmd_vel を出し、台車をその場で回したり、決めた距離だけ動かしたりする共通部分。

drive_pattern（評価）と table_dock（机へ寄せる）が使う。

安全の約束:
- 区間ごとに時間の上限を置き、超えたら止めて例外にする
- /odom か /wheel/odom が stale_s 秒届かなければ速度 0 を出して待ち、stale_abort_s 秒を超えたら例外にする
  （WiFi が一瞬途切れるたびに走行を終えないため。待った分だけ時間の上限を延ばす）
"""

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from xlerobot_nav.msgs import Unwrapper, quaternion_to_yaw, to_seconds


class Track:
    """1 つのオドメトリのトピックの最新の値と、巻き戻した yaw。"""

    def __init__(self) -> None:
        self.msg: Odometry | None = None
        self.t_rx = 0.0
        self.yaw = Unwrapper()

    def update(self, msg: Odometry) -> None:
        self.msg = msg
        self.t_rx = time.monotonic()
        self.yaw.update(quaternion_to_yaw(msg.pose.pose.orientation))

    def xy(self) -> tuple[float, float]:
        assert self.msg is not None
        p = self.msg.pose.pose.position
        return p.x, p.y

    def snapshot(self) -> dict:
        assert self.msg is not None
        p, q = self.msg.pose.pose.position, self.msg.pose.pose.orientation
        return {
            "t": to_seconds(self.msg.header.stamp),
            "p": [p.x, p.y, p.z],
            "q": [q.x, q.y, q.z, q.w],
            "yaw_unwrapped": self.yaw.value,
        }


class OdomDriver(Node):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.speed = 0.1
        self.turn_speed = 20.0
        self.stale_s = 0.5
        self.stale_abort_s = 2.0
        arkit_topic = str(self.declare_parameter("arkit_odom_topic", "/pocketsensor/odom").value)
        self.tracks = {"odom": Track(), "wheel": Track(), "arkit": Track()}
        self.create_subscription(Odometry, "odom", self.tracks["odom"].update, 50)
        self.create_subscription(Odometry, "wheel/odom", self.tracks["wheel"].update, 50)
        self.create_subscription(Odometry, arkit_topic, self.tracks["arkit"].update, 50)
        self.cmd = self.create_publisher(Twist, "cmd_vel", 10)

    def send(self, vx: float = 0.0, wz_dps: float = 0.0, vy: float = 0.0) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = math.radians(wz_dps)
        self.cmd.publish(msg)

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def hold(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send()
            self.spin_for(0.05)

    def fresh(self) -> bool:
        return all(
            t.msg is not None and time.monotonic() - t.t_rx < self.stale_s
            for k, t in self.tracks.items()
            if k != "arkit"
        )

    def wait_inputs(self, timeout: float = 10.0) -> None:
        """3 つのオドメトリがそろうまで待つ。購読が DDS でつながるまで、1 秒以上かかることがある。"""
        deadline = time.monotonic() + timeout
        while not (all(t.msg is not None for t in self.tracks.values()) and self.fresh()):
            if time.monotonic() > deadline:
                raise RuntimeError("odom、wheel/odom、ARKit の姿勢のどれかが届かない")
            self.spin_for(0.1)

    def _ensure_fresh(self) -> float:
        """途切れていれば速度 0 を出して待ち、待った秒数を返す。"""
        if self.fresh():
            return 0.0
        t0 = time.monotonic()
        while not self.fresh():
            if time.monotonic() - t0 > self.stale_abort_s:
                raise RuntimeError("odom か wheel/odom が途切れた")
            self.send()
            self.spin_for(0.05)
        return time.monotonic() - t0

    def turn(self, target_deg: float) -> None:
        odom = self.tracks["odom"]
        start = odom.yaw.value
        sign = 1.0 if target_deg > 0 else -1.0
        deadline = time.monotonic() + abs(target_deg) / self.turn_speed * 3.0 + 5.0
        while True:
            deadline += self._ensure_fresh()
            if time.monotonic() > deadline:
                raise RuntimeError("旋回が時間内に終わらない")
            remaining = abs(target_deg) - sign * math.degrees(odom.yaw.value - start)
            if remaining < 0.3:
                break
            self.send(wz_dps=sign * min(self.turn_speed, max(3.0, remaining)))
            self.spin_for(0.05)
        self.hold(1.0)

    def translate(
        self, distance: float, sideways: bool = False, *, min_speed: float = 0.02, hold_heading: bool = False
    ) -> float:
        """始めの向きの前（sideways なら左）へ distance [m] 進み、実際に進んだ距離を返す。負なら後ろ（右）。

        hold_heading なら、進むあいだの向きの変化を旋回で打ち消す。横移動で台車が勝手に回るのを抑えるため。
        """
        odom = self.tracks["odom"]
        x0, y0 = odom.xy()
        h0 = odom.yaw.value
        axis = (-math.sin(h0), math.cos(h0)) if sideways else (math.cos(h0), math.sin(h0))
        sign = 1.0 if distance > 0 else -1.0
        deadline = time.monotonic() + abs(distance) / self.speed * 3.0 + 5.0

        def done() -> float:
            x, y = odom.xy()
            return (x - x0) * axis[0] + (y - y0) * axis[1]

        while True:
            deadline += self._ensure_fresh()
            if time.monotonic() > deadline:
                raise RuntimeError("並進が時間内に終わらない")
            remaining = abs(distance) - sign * done()
            if remaining < 0.002:
                break
            # 残りが小さくなったら、残り [m] に比例して落とす（下限 min_speed）
            v = sign * min(self.speed, max(min_speed, remaining))
            # 向きの誤差 1° につき 1 deg/s で戻す（上限 5 deg/s）。融合した yaw の遅れ（最長 0.3 秒）より十分遅い
            wz = max(-5.0, min(5.0, math.degrees(h0 - odom.yaw.value))) if hold_heading else 0.0
            if sideways:
                self.send(vy=v, wz_dps=wz)
            else:
                self.send(vx=v, wz_dps=wz)
            self.spin_for(0.05)
        self.hold(1.0)
        return done()
