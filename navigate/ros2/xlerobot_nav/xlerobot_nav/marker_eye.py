"""iPad のカラーで壁のマーカーを、深度でその壁を測る。goto_place が着いたあとの合わせ込みに使う。

マーカーの中心は PnP で求め、何枚かの中央値を取る。向きはマーカーからは取らず、壁の点の当てはめ（navgate.marker_align）で求める。
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from navgate.marker_align import MarkerRef, wall_slope
from navgate.table_edge import points_from_depth
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import Buffer, TransformException, TransformListener

from xlerobot_nav.msgs import transform_to_matrix

COLOR = "/pocketsensor/color"
DEPTH = "/pocketsensor/depth"


class MarkerEye:
    def __init__(self, node: Node, base_frame: str = "base_link", size_m: float = 0.035) -> None:
        self.node = node
        self.base_frame = base_frame
        self.size_m = size_m
        self.color: list[CompressedImage] = []
        self.depth: list[Image] = []
        self.color_info: CameraInfo | None = None
        self.depth_info: CameraInfo | None = None
        node.create_subscription(
            CompressedImage, f"{COLOR}/image/compressed", self._on_color, qos_profile_sensor_data
        )
        node.create_subscription(Image, f"{DEPTH}/image", self._on_depth, qos_profile_sensor_data)
        node.create_subscription(
            CameraInfo, f"{COLOR}/camera_info", self._on_color_info, qos_profile_sensor_data
        )
        node.create_subscription(
            CameraInfo, f"{DEPTH}/camera_info", self._on_depth_info, qos_profile_sensor_data
        )
        self.tf = Buffer()
        self._listener = TransformListener(self.tf, node)

    def _on_color(self, msg: CompressedImage) -> None:
        self.color = [msg]

    def _on_depth(self, msg: Image) -> None:
        self.depth = [msg]

    def _on_color_info(self, msg: CameraInfo) -> None:
        self.color_info = msg

    def _on_depth_info(self, msg: CameraInfo) -> None:
        self.depth_info = msg

    def _base_from(self, frame: str) -> np.ndarray | None:
        try:
            return transform_to_matrix(self.tf.lookup_transform(self.base_frame, frame, Time()))
        except TransformException:
            return None

    def _marker_center(self, marker_id: int) -> np.ndarray | None:
        """最新のカラー 1 枚から、base_link のマーカーの中心 (x, y, z)。"""
        if not self.color or self.color_info is None:
            return None
        msg, info = self.color[0], self.color_info
        self.color = []
        gray = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or marker_id not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2)
        half = self.size_m / 2.0
        obj = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64
        )
        ok, _, tvec = cv2.solvePnP(
            obj, c, np.array(info.k).reshape(3, 3), np.array(info.d), flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        T = self._base_from(msg.header.frame_id or info.header.frame_id)
        if not ok or T is None:
            return None
        return T[:3, :3] @ tvec.ravel() + T[:3, 3]

    def _wall_yaw(self, center: np.ndarray) -> float | None:
        if not self.depth or self.depth_info is None:
            return None
        msg, info = self.depth[0], self.depth_info
        self.depth = []
        d = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
        d[d == 0] = np.nan
        T = self._base_from(info.header.frame_id)
        if T is None:
            return None
        points = points_from_depth(d / 1000.0, (info.k[0], info.k[4], info.k[2], info.k[5]), T)
        return wall_slope(points, marker_xy=(float(center[0]), float(center[1])), marker_z=float(center[2]))

    def measure(
        self, marker_id: int, frames: int = 10, timeout: float = 15.0
    ) -> tuple[np.ndarray, float] | None:
        """マーカーの中心 (x, y, z) と壁に対する向き [rad] の、frames 枚の中央値。見えなければ None。"""
        centers: list[np.ndarray] = []
        yaws: list[float] = []
        deadline = time.monotonic() + timeout
        while (len(centers) < frames or len(yaws) < frames) and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if len(centers) < frames:
                c = self._marker_center(marker_id)
                if c is not None:
                    centers.append(c)
            if centers and len(yaws) < frames:
                y = self._wall_yaw(np.median(centers, axis=0))
                if y is not None:
                    yaws.append(y)
        if len(centers) < frames or len(yaws) < frames:
            return None
        return np.median(centers, axis=0), float(np.median(yaws))

    def reference(self, marker_id: int, frames: int = 20) -> MarkerRef | None:
        got = self.measure(marker_id, frames=frames)
        if got is None:
            return None
        center, yaw = got
        return MarkerRef(
            marker_id=marker_id,
            marker_xy=(round(float(center[0]), 4), round(float(center[1]), 4)),
            wall_yaw=round(yaw, 5),
        )
