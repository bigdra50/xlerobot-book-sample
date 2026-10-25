"""保存した地図を、可視化ツールへ出す。view.launch.py が起動する。

    ros2 run xlerobot_nav saved_map --ros-args -p cloud:=/tmp/xlerobot_view_cloud.ply

- 色つきの点群: rtabmap-export で書き出した PLY を、/cloud_map_dense（map の座標）へ TRANSIENT_LOCAL で出す
- 占有格子と通った道: 地図の写しを読んだ rtabmap（publish_map_service）に、/map と通った道を 1 回だけ出させる。
  publish_map_service を空にすると、点群だけを出す（走らせているスタックに重ねるとき）
  - rtabmap は購読者がいないと出さないので、どちらもこのノードが購読しておく
  - rtabmap の通った道は TRANSIENT_LOCAL ではなく、後から来た購読者へ届かない。
    view.launch.py が /mapPath_rtabmap へ逃がし、このノードが /mapPath へ TRANSIENT_LOCAL で出し直す
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from navgate.ply import interleave_xyzrgb, pack_rgb, read_ply_vertices
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.srv import PublishMap
from sensor_msgs.msg import PointCloud2, PointField

MAX_ATTEMPTS = 10
LATCHED = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)


def cloud_message(path: Path, frame_id: str, stamp) -> PointCloud2:
    v = read_ply_vertices(path.read_bytes())
    n = len(v["x"])
    fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(("x", "y", "z", "rgb"))
    ]
    msg = PointCloud2(height=1, width=n, fields=fields, is_bigendian=False, point_step=16, row_step=16 * n)
    msg.is_dense = True
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.data = interleave_xyzrgb(v["x"], v["y"], v["z"], pack_rgb(v["red"], v["green"], v["blue"]))
    return msg


class SavedMap(Node):
    def __init__(self) -> None:
        super().__init__("saved_map")
        self.declare_parameter("cloud", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_map_service", "/rtabmap/publish_map")
        cloud = self.get_parameter("cloud").value
        if cloud:
            self.cloud_pub = self.create_publisher(PointCloud2, "/cloud_map_dense", LATCHED)
            msg = cloud_message(
                Path(cloud), self.get_parameter("frame_id").value, self.get_clock().now().to_msg()
            )
            self.cloud_pub.publish(msg)
            self.get_logger().info(f"{msg.width} 点を /cloud_map_dense に出した（{cloud}）")
        service = self.get_parameter("publish_map_service").value
        if not service:
            return
        self.path_pub = self.create_publisher(PathMsg, "/mapPath", LATCHED)
        self.received: set[str] = set()
        self.create_subscription(OccupancyGrid, "/map", self._on_map, LATCHED)
        self.create_subscription(PathMsg, "/mapPath_rtabmap", self._on_path, 10)
        self.client = self.create_client(PublishMap, service)
        self.attempts = 0
        self.pending = None
        self.create_timer(1.0, self._tick)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.received.add("map")
        self.get_logger().info(
            f"/map {msg.info.width}x{msg.info.height}（{msg.info.resolution:.2f} m）", once=True
        )

    def _on_path(self, msg: PathMsg) -> None:
        self.received.add("path")
        self.path_pub.publish(msg)
        self.get_logger().info(f"通った道 {len(msg.poses)} 点を /mapPath に出した", once=True)

    def _tick(self) -> None:
        if {"map", "path"} <= self.received or self.attempts >= MAX_ATTEMPTS:
            return
        if not self.client.service_is_ready() or (self.pending is not None and not self.pending.done()):
            return
        # 起動した直後は、rtabmap がこのノードの購読をまだ見つけていないことがあるので、届くまで出させ直す
        self.pending = self.client.call_async(
            PublishMap.Request(global_map=True, optimized=True, graph_only=False)
        )
        self.attempts += 1
        if self.attempts == MAX_ATTEMPTS:
            self.get_logger().warning(f"{MAX_ATTEMPTS} 回出させても届かなかった: {self.received}")


def main() -> None:
    rclpy.init()
    node = SavedMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
