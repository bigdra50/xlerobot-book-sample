"""本書の host の観測を、ROS 2 の話題に出す。見るためと記録のためのもので、推論はこれを通さない。

    ros2 run xlerobot_nav host_bridge --ros-args -p host:=<Pi の IP> -p port:=5557

host を --observe.port=5557 で起動しておく（host/xlerobot_host.py）。
公式の観測の口（5556、PUSH）へはつながない。つないだ相手が複数いると、観測が振り分けられて方策の client の分が減るため。

出す話題:
- /xlerobot/<カメラ名>/image/compressed（sensor_msgs/CompressedImage、JPEG をそのまま）
- /xlerobot/joint_positions（sensor_msgs/JointState）。position は host の値のまま（キャリブレーションの範囲を −100〜100 にした値、gripper は 0〜100）で、rad ではない
- /joint_states（sensor_msgs/JointState）。公式の URDF の関節名と角度 [rad]（navgate.joint_map）。robot_state_publisher が腕と head の TF を出す
  換算には、キャリブレーションのファイル（calibration）が要る。無ければ /joint_states は出さない

head カメラの画像の frame_id は、URDF の head_camera_rgb_optical_frame にする。

host の観測は時刻を持たないので、stamp は受け取った時刻にする。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import rclpy
import zmq
from navgate.joint_map import to_urdf
from navgate.observation import split_observation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState

# カメラ名から URDF のリンク。手首のカメラは、公式の URDF では光学 frame ではなくカメラの取り付けのリンク
FRAMES = {
    "head": "head_camera_rgb_optical_frame",
    "left_wrist": "Left_Arm_Camera",
    "right_wrist": "Right_Arm_Camera",
}


class HostBridge(Node):
    def __init__(self) -> None:
        super().__init__("host_bridge")
        host = str(self.declare_parameter("host", "xlerobot-pi.local").value)
        port = int(self.declare_parameter("port", 5557).value)
        # host のキャリブレーション JSON（Pi の ~/.cache/huggingface/lerobot/calibration/robots/xlerobot/<robot.id>.json）を
        # navigate/gate/runs/calibration/ へ写しておく。コンテナからは /book/navigate/gate/runs/calibration/ に見える
        cal_path = Path(
            str(
                self.declare_parameter(
                    "calibration", "/book/navigate/gate/runs/calibration/follower_xlerobot.json"
                ).value
            )
        )
        self.cal = json.loads(cal_path.read_text()) if cal_path.exists() else None
        if self.cal is None:
            self.get_logger().warn(f"no calibration at {cal_path}; /joint_states is not published")
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(f"tcp://{host}:{port}")
        self.image_pubs: dict[str, rclpy.publisher.Publisher] = {}
        self.joint_pub = self.create_publisher(JointState, "/xlerobot/joint_positions", 10)
        self.urdf_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.count = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.get_logger().info(f"subscribing to tcp://{host}:{port}")

    def _loop(self) -> None:
        while not self.stop.is_set():
            if not self.sub.poll(500):
                continue
            try:
                obs = json.loads(self.sub.recv_string())
            except (json.JSONDecodeError, zmq.ZMQError):
                continue
            self._publish(obs)

    def _publish(self, obs: dict) -> None:
        images, joints, _base = split_observation(obs)
        stamp = self.get_clock().now().to_msg()
        for name, jpeg in images.items():
            pub = self.image_pubs.get(name)
            if pub is None:
                pub = self.create_publisher(
                    CompressedImage, f"/xlerobot/{name}/image/compressed", qos_profile_sensor_data
                )
                self.image_pubs[name] = pub
            msg = CompressedImage(format="jpeg", data=jpeg)
            msg.header.stamp = stamp
            msg.header.frame_id = FRAMES.get(name, f"{name}_camera")
            pub.publish(msg)
        if joints:
            js = JointState(name=[n for n, _ in joints], position=[v for _, v in joints])
            js.header.stamp = stamp
            self.joint_pub.publish(js)
            if self.cal is not None:
                q = to_urdf(dict(joints), self.cal)
                urdf = JointState(name=list(q), position=list(q.values()))
                urdf.header.stamp = stamp
                self.urdf_pub.publish(urdf)
        self.count += 1
        if self.count == 1:
            self.get_logger().info(f"first observation: cameras {sorted(images)}, {len(joints)} joints")

    def destroy_node(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)
        self.sub.close(linger=0)
        self.ctx.term()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = HostBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
