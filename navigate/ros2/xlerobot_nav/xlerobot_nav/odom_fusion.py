"""/wheel/odom と ARKit の姿勢（/pocketsensor/odom）を融合し、/odom と tf の odom -> base_link を出す。

融合の規則は navgate.fusion（その場の旋回は車輪、並進のあいだの回転は ARKit）。
ARKit はトラッキングが normal のあいだだけ使い、world の原点が作り直された区間は使わない。
車輪が止まっているのに ARKit で台車が動いたとき（手で押された、床を滑った）は、その動きを ARKit から取る。
そのために、URDF の取り付け（base_link から pocketsensor_link）を TF から一度だけ読む。
"""

from __future__ import annotations

from collections import deque

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from navgate.fusion import FusionParams, Pose2D
from navgate.fusion_stream import FusionStream, body_delta
from pocketsensor_msgs.msg import TrackingStatus
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from xlerobot_nav.msgs import Unwrapper, quaternion_to_yaw, to_seconds, to_stamp, yaw_to_quaternion


class OdomFusion(Node):
    def __init__(self) -> None:
        super().__init__("odom_fusion")
        defaults = FusionParams()
        params = FusionParams(
            arkit_rot_scale_err=float(
                self.declare_parameter("arkit_rot_scale_err", defaults.arkit_rot_scale_err).value
            ),
            wheel_yaw_err_per_m_fwd=float(
                self.declare_parameter("wheel_yaw_err_per_m_fwd", defaults.wheel_yaw_err_per_m_fwd).value
            ),
            wheel_yaw_err_per_m_side=float(
                self.declare_parameter("wheel_yaw_err_per_m_side", defaults.wheel_yaw_err_per_m_side).value
            ),
        )
        self.stream = FusionStream(params, max_delay_s=float(self.declare_parameter("max_delay", 0.3).value))
        self.lens_frame = str(self.declare_parameter("lens_frame", "pocketsensor_link").value)
        self._mount_known = False
        self._pushed = 0
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.odom_frame = str(self.declare_parameter("odom_frame", "odom").value)
        self.base_frame = str(self.declare_parameter("base_frame", "base_link").value)
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        arkit_topic = str(self.declare_parameter("arkit_odom_topic", "/pocketsensor/odom").value)
        tracking_topic = str(self.declare_parameter("tracking_topic", "/pocketsensor/tracking").value)
        self.pub = self.create_publisher(Odometry, "odom", 50)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "wheel/odom", self._on_wheel, 100)
        self.create_subscription(Odometry, arkit_topic, self._on_arkit, 100)
        self.create_subscription(TrackingStatus, tracking_topic, self._on_tracking, 100)
        self._wheel_yaw = Unwrapper()
        self._tracking_ok = False
        self._epoch = -1
        self._last: tuple[float, Pose2D] | None = None
        self._weights: deque[float] = deque(maxlen=500)
        self.create_timer(0.02, self._tick)
        self.create_timer(1.0, self._read_mount)
        self.create_timer(10.0, self._report)

    def _on_wheel(self, msg: Odometry) -> None:
        p = msg.pose.pose
        yaw = self._wheel_yaw.update(quaternion_to_yaw(p.orientation))
        self.stream.feed_wheel(to_seconds(msg.header.stamp), Pose2D(p.position.x, p.position.y, yaw))

    def _on_tracking(self, msg: TrackingStatus) -> None:
        self._tracking_ok = msg.state == TrackingStatus.STATE_NORMAL
        self._epoch = int(msg.origin_epoch)

    def _read_mount(self) -> None:
        if self._mount_known:
            return
        try:
            tf = self._tf_buffer.lookup_transform(self.base_frame, self.lens_frame, Time())
        except Exception:  # noqa: BLE001  TF がまだ届いていない。次の周期で読み直す
            return
        t = tf.transform
        self.stream.lens_xy = (t.translation.x, t.translation.y)
        self.stream.mount_yaw = quaternion_to_yaw(t.rotation)
        self._mount_known = True
        self.get_logger().info(
            f"mount: lens ({t.translation.x:.3f}, {t.translation.y:.3f}) m, yaw {self.stream.mount_yaw:.4f} rad"
        )

    def _on_arkit(self, msg: Odometry) -> None:
        p = msg.pose.pose
        yaw = quaternion_to_yaw(p.orientation)
        xy = (p.position.x, p.position.y) if self._mount_known else (None, None)
        self.stream.feed_arkit(
            to_seconds(msg.header.stamp), yaw, ok=self._tracking_ok, epoch=self._epoch, x=xy[0], y=xy[1]
        )

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        for t, pose, weight in self.stream.ready(now):
            self._weights.append(weight)
            if self.stream.last_pushed:
                self._pushed += 1
            self._publish(t, pose)

    def _publish(self, t: float, pose: Pose2D) -> None:
        msg = Odometry()
        msg.header.stamp = to_stamp(t)
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        msg.pose.pose.orientation = yaw_to_quaternion(pose.theta)
        if self._last is not None:
            t0, p0 = self._last
            dt = max(1e-3, t - t0)
            dx, dy, dth = body_delta(p0, pose)
            msg.twist.twist.linear.x = dx / dt
            msg.twist.twist.linear.y = dy / dt
            msg.twist.twist.angular.z = dth / dt
        self._last = (t, pose)
        self.pub.publish(msg)
        if self.publish_tf:
            tf = TransformStamped()
            tf.header = msg.header
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = pose.x
            tf.transform.translation.y = pose.y
            tf.transform.rotation = msg.pose.pose.orientation
            self.tf.sendTransform(tf)

    def _report(self) -> None:
        if self._weights:
            used = sum(1 for w in self._weights if w > 0.5) / len(self._weights)
            self.get_logger().info(
                f"arkit weight > 0.5 in {used * 100:.0f}% of the last {len(self._weights)} steps, "
                f"tracking {'normal' if self._tracking_ok else 'not normal'}, epoch {self._epoch}, "
                f"pushed steps {self._pushed}"
            )


def main() -> None:
    rclpy.init()
    node = OdomFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
