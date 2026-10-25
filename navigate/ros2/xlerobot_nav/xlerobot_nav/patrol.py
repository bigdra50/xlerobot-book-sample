"""残した地図の上で、台車に巡回させる。map.launch.py を localize:=true で起動した状態で使う。

    ros2 run xlerobot_nav patrol --ros-args -p laps:=2 -p extra:='[[-0.03, -0.62]]' -p out:=/out/patrol/run.json

巡回の点は、地図の通った道から、地図を作ったときにその場で 1 周回れた場所を取り出して決める（navgate.patrol）。
extra で点を足す。waypoints を渡せば、その点だけを使う（どちらも [[x, y], ...]、map の座標）。
点は、地図の上で障害物か未知のマスまで min_clearance [m] 以上あるものだけを使う。
台車に近い点から巡回し、各点では次の点の方を向いて dwell 秒止まる。
巡回の点は /patrol/waypoints に出す。円は、その点で回るときに台車の角が通る範囲。
dry_run:=true では台車を動かさず、巡回の順を決めて /patrol/waypoints に出したまま待つ（Ctrl+C で終わる）。
台車の位置は TF から引き、無ければ start（[x, y]）を使う。保存した地図の表示（view.launch.py）で順を確かめられる。

安全の約束:
- rtabmap が地図の上の位置を取り直す（/localization_pose が届く）まで動かさない。wait_localization 秒で届かなければ終わる
- 目標ごとに goal_timeout 秒の上限を置き、超えたら取り消す
- 目標に届かなかったら、巡回をやめる。止め方は Nav2 に任せる（速度 0）
- Ctrl+C でも、送っている目標を取り消してから終わる
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from navgate.patrol import clearance_at, loop_route, parse_points, turn_spots
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rtabmap_msgs.srv import PublishMap
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from xlerobot_nav.msgs import quaternion_to_yaw, yaw_to_quaternion

LATCHED = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)
# 台車の外形（0.35 x 0.45 m）の対角の半分。その場で回るとき、角はこの半径の円を通る
TURN_RADIUS = 0.285


class Patrol(Node):
    def __init__(self) -> None:
        super().__init__("patrol")
        self.declare_parameter("laps", 2)
        self.declare_parameter("dwell", 2.0)
        self.declare_parameter("goal_timeout", 120.0)
        self.declare_parameter("min_clearance", 0.33)
        self.declare_parameter("wait_localization", 60.0)
        # 点は JSON の文字列でも数の配列でも受ける（-p extra:='[[-0.03, -0.62]]' は数の配列として届く）
        any_type = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("waypoints", "", any_type)
        self.declare_parameter("extra", "", any_type)
        self.declare_parameter("out", "")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("start", "", any_type)
        self.grid: OccupancyGrid | None = None
        self.path: PathMsg | None = None
        self.localized = False
        self.create_subscription(OccupancyGrid, "/map", lambda m: setattr(self, "grid", m), LATCHED)
        # rtabmap の通った道は TRANSIENT_LOCAL ではない
        self.create_subscription(PathMsg, "/mapPath", lambda m: setattr(self, "path", m), 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/localization_pose", lambda _m: setattr(self, "localized", True), 10
        )
        self.markers = self.create_publisher(MarkerArray, "/patrol/waypoints", LATCHED)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.publish_map = self.create_client(PublishMap, "/rtabmap/publish_map")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.active_goal = None

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, done, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if done():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return done()

    def pose(self) -> tuple[float, float, float] | None:
        try:
            t = self.tf.lookup_transform("map", "base_link", Time())
        except Exception:  # noqa: BLE001  TF が揃うまでは引けない
            return None
        return (t.transform.translation.x, t.transform.translation.y, quaternion_to_yaw(t.transform.rotation))

    def points(self) -> list[tuple[float, float]]:
        given = parse_points(self.get_parameter("waypoints").value)
        if given:
            return given
        if not self.wait_for(lambda: self.path is not None, 5.0):
            # 位置の取り直しのモードでは、地図が更新されるまで通った道が出ないことがある
            if self.publish_map.wait_for_service(timeout_sec=5.0):
                self.publish_map.call_async(
                    PublishMap.Request(global_map=True, optimized=True, graph_only=True)
                )
            if not self.wait_for(lambda: self.path is not None, 10.0):
                raise RuntimeError("/mapPath（地図の通った道）が届かない")
        poses = [
            (p.pose.position.x, p.pose.position.y, quaternion_to_yaw(p.pose.orientation))
            for p in self.path.poses
        ]
        return turn_spots(poses) + parse_points(self.get_parameter("extra").value)

    def usable(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        g = self.grid
        grid = np.array(g.data, np.int16).reshape(g.info.height, g.info.width)
        origin = (g.info.origin.position.x, g.info.origin.position.y)
        need = self.get_parameter("min_clearance").value
        kept = []
        for x, y in points:
            c = clearance_at(grid, g.info.resolution, origin, x, y)
            self.get_logger().info(
                f"点 ({x:+.2f}, {y:+.2f}): 余裕 {c:.2f} m" + ("" if c >= need else " -> 使わない")
            )
            if c >= need:
                kept.append((x, y))
        return kept

    def show(self, route: list[tuple[float, float, float]]) -> None:
        stamp = self.get_clock().now().to_msg()
        out = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for k, (x, y, yaw) in enumerate(route):
            for n, (kind, z, scale, rgba) in enumerate(
                (
                    (Marker.CYLINDER, 0.01, (2 * TURN_RADIUS, 2 * TURN_RADIUS, 0.02), (0.2, 0.9, 0.4, 0.25)),
                    (Marker.ARROW, 0.05, (0.35, 0.05, 0.05), (0.2, 0.9, 0.4, 1.0)),
                    (Marker.TEXT_VIEW_FACING, 0.45, (0.0, 0.0, 0.15), (1.0, 1.0, 1.0, 1.0)),
                )
            ):
                m = Marker(ns="patrol", id=3 * k + n, type=kind, action=Marker.ADD)
                m.header.frame_id, m.header.stamp = "map", stamp
                m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
                m.pose.orientation = yaw_to_quaternion(yaw)
                m.scale.x, m.scale.y, m.scale.z = scale
                m.color.r, m.color.g, m.color.b, m.color.a = rgba
                m.text = str(k + 1)
                out.markers.append(m)
        self.markers.publish(out)

    def go(self, x: float, y: float, yaw: float) -> dict:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation = yaw_to_quaternion(yaw)
        record = {"goal": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)]}
        t0 = time.monotonic()
        send = self.nav.send_goal_async(goal)
        self.wait_for(send.done, 10.0)
        handle = send.result() if send.done() else None
        if handle is None or not handle.accepted:
            return {**record, "status": "rejected"}
        self.active_goal = handle
        result = handle.get_result_async()
        if not self.wait_for(result.done, self.get_parameter("goal_timeout").value):
            self.cancel()
            return {**record, "status": "timeout", "seconds": round(time.monotonic() - t0, 1)}
        self.active_goal = None
        status = result.result().status
        record.update(
            status="succeeded" if status == GoalStatus.STATUS_SUCCEEDED else f"status {status}",
            seconds=round(time.monotonic() - t0, 1),
        )
        now = self.pose()
        if now is not None:
            record["error_m"] = round(math.hypot(now[0] - x, now[1] - y), 3)
            record["error_deg"] = round(math.degrees(math.remainder(now[2] - yaw, math.tau)), 1)
        return record

    def cancel(self) -> None:
        if self.active_goal is not None:
            future = self.active_goal.cancel_goal_async()
            self.wait_for(future.done, 5.0)
            self.active_goal = None

    def run(self) -> int:
        summary: dict = {"laps": []}
        try:
            if not self.wait_for(lambda: self.grid is not None, 30.0):
                raise RuntimeError(
                    "/map（占有格子）が届かない。map.launch.py を localize:=true で起動しておく"
                )
            points = self.usable(self.points())
            if not points:
                raise RuntimeError("使える巡回の点が無い")
            dry_run = self.get_parameter("dry_run").value
            wait = 0 if dry_run else self.get_parameter("wait_localization").value
            if wait > 0:
                self.get_logger().info(f"rtabmap が地図の上の位置を取り直すのを、{wait:.0f} 秒まで待つ")
                if not self.wait_for(lambda: self.localized, wait):
                    raise RuntimeError(
                        "地図の上の位置を取り直せなかった。台車を地図の中の見覚えのある向きへ向け直す"
                    )
            start = parse_points(self.get_parameter("start").value)
            if self.wait_for(lambda: self.pose() is not None, 10.0):
                x0, y0, _ = self.pose()
            elif dry_run and start:
                x0, y0 = start[0]
            else:
                raise RuntimeError("map -> base_link を引けない")
            route = loop_route((x0, y0), points)
            summary["route"] = [[round(x, 3), round(y, 3), round(math.degrees(a), 1)] for x, y, a in route]
            self.show(route)
            self.get_logger().info(f"巡回の順: {summary['route']}")
            if dry_run:
                summary["result"] = "dry_run"
                self.write(summary)
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.5)
                return 0
            if not self.nav.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("Nav2 の navigate_to_pose が無い")
            for lap in range(self.get_parameter("laps").value):
                records = []
                summary["laps"].append(records)
                for k, (x, y, yaw) in enumerate(route):
                    self.get_logger().info(f"{lap + 1} 周目、点 {k + 1} へ ({x:+.2f}, {y:+.2f})")
                    record = self.go(x, y, yaw)
                    records.append(record)
                    self.get_logger().info(f"  {record}")
                    if record["status"] != "succeeded":
                        raise RuntimeError(f"点 {k + 1} に届かなかった: {record['status']}")
                    self.spin_for(self.get_parameter("dwell").value)
            summary["result"] = "completed"
            return 0
        except KeyboardInterrupt:
            self.cancel()
            summary.setdefault("result", "interrupted")
            return 130
        except RuntimeError as e:
            self.cancel()
            summary["result"] = f"stopped: {e}"
            self.get_logger().error(str(e))
            return 1
        finally:
            self.write(summary)

    def write(self, summary: dict) -> None:
        out = self.get_parameter("out").value
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=1)


def main() -> None:
    rclpy.init()
    node = Patrol()
    code = node.run()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
