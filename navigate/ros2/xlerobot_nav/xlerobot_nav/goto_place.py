"""名前を付けた場所へ行く。map.launch.py を localize:=true で起動した状態で使う。

    ros2 run xlerobot_nav goto_place --ros-args -p place:=desk_side
    ros2 run xlerobot_nav goto_place --ros-args -p leave:=true          # 場所から待機位置へ出るだけ
    ros2 run xlerobot_nav goto_place --ros-args -p adjust:=desk_side -p nudge:='[0.01, -0.02, 0.0]'
    ros2 run xlerobot_nav goto_place --ros-args -p save:=desk_side -p entry:='[-0.03, -0.64, 0.0]' -p standoff:=0.15
    ros2 run xlerobot_nav goto_place --ros-args -p save_marker:=desk_side -p marker_id:=48
    ros2 run xlerobot_nav goto_place --ros-args -p align_only:=desk_side   # 地図を使わず、マーカーだけで合わせる

場所は places_file（JSON）に、最終の位置（pose）と待機位置（entry）で持つ。考え方は navgate.places にある。
save を渡すと、いまの地図の上の位置を pose として登録して終わる（台車は動かさない）。
adjust を渡すと、その場所に着いている台車を nudge（前 m、左 m、左回り deg）だけ動かし、動いた先を pose として登録し直す。
待機位置と standoff は変えない。1 回に動かせるのは前後と左右 0.10 m、向き 10 度まで。
align_only を渡すと、地図の上の位置を使わず、見えている壁のマーカーだけでその場所に合わせ込み、
合わせた位置を地図の上の位置として決め直す。地図の上の位置を見失ったときに、場所の前まで手で動かしてから使う。
save_marker を渡すと、いまその場所にいる台車から見た壁のマーカーを測り、場所の基準として登録する（台車は動かさない）。

行き方:
0. 登録した場所のそば（near_radius 以内）にいれば、入った手順を逆にたどって、その場所の待機位置へ出る
1. 位置の取り直しを再開し、/localization_pose が届くまで待つ。届かなければ、その場で search_step 度ずつ回って探す
2. 待機位置まで Nav2 で行く
3. /scan を地図の占有格子に重ね、推定の位置が合っているかを確かめる。ずれが大きければ寄せずに止める
   位置の取り直しを止め、navgate.places.next_step の 1 手ずつで最終の位置まで寄せる
3b. 場所にマーカーの基準があれば、マーカーの中心と壁の向きを見て、基準と同じ見え方になるまで合わせ込む。
    合わせ込めたら、台車は登録した位置にいるので、その位置を /initialpose で rtabmap に渡し、地図の上の位置を決め直す
4. 着いたら止まったままにする。位置の取り直しは止めたまま（机の近くで取り直すと位置が跳ぶため）。
   次に goto_place を起動したとき、最初に再開する

安全の約束:
- 位置が取り直せないまま、Nav2 を使わない
- 寄せる手は max_steps 回まで。1 手ごとに地図の上の位置を読み直して次を決める
- 探すための旋回は、周りの空きを確かめない。回っても当たらない場所で使う
- マーカーの合わせ込みは、1 手で前後と左右 MAX_ALIGN_M、向き MAX_ALIGN_DEG を超えるなら止める（見間違いか、地図の大きなずれ）
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from navgate.marker_align import MarkerRef, align_step
from navgate.places import Place, exit_of, load_places, near, next_step, save_place
from navgate.scan_fit import free_hit_ratio
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from xlerobot_nav.marker_eye import MarkerEye
from xlerobot_nav.msgs import quaternion_to_yaw, yaw_to_quaternion
from xlerobot_nav.odom_driver import OdomDriver

# 場所は部屋ごとのデータなので、地図のデータベース（/out/rtabmap.db）と同じ場所に置く
DEFAULT_PLACES = "/out/places.json"
# マーカーで直す量の上限。地図だけで寄せた位置は、机のそばで 6〜11 cm ずれることがあった
MAX_ALIGN_M = 0.20
MAX_ALIGN_DEG = 15.0
# 寄せる前の確かめで許す、/scan の点が地図の空きのマスに落ちる割合。正しい推定でも縁のにじみで数 % は落ちる
MAX_FREE_HIT = 0.15


class GotoPlace(OdomDriver):
    def __init__(self) -> None:
        super().__init__("goto_place")

        def param(name: str, default):
            return self.declare_parameter(name, default).value

        self.place_name = str(param("place", ""))
        self.save_name = str(param("save", ""))
        self.entry = [float(v) for v in param("entry", [0.0, 0.0, 0.0])]
        self.standoff = float(param("standoff", 0.15))
        self.places_file = str(param("places_file", DEFAULT_PLACES))
        self.wait_localization = float(param("wait_localization", 15.0))
        self.search_step = float(param("search_step", 30.0))
        self.search_turns = int(param("search_turns", 12))
        self.goal_timeout = float(param("goal_timeout", 120.0))
        self.max_steps = int(param("max_steps", 15))
        self.leave_only = bool(param("leave", False))
        self.adjust_name = str(param("adjust", ""))
        self.nudge = [float(v) for v in param("nudge", [0.0, 0.0, 0.0])]
        self.save_marker = str(param("save_marker", ""))
        self.align_only = str(param("align_only", ""))
        self.marker_id = int(param("marker_id", 48))
        self.align_tries = int(param("align_tries", 4))
        self.eye = MarkerEye(self, size_m=float(param("marker_size", 0.035)))
        self.near_radius = float(param("near_radius", 0.3))
        self.speed = float(param("speed", 0.05))
        self.turn_speed = float(param("turn_speed", 12.0))
        self.out = str(param("out", "/out/places/last.json"))
        self.localized_at = 0.0
        self.create_subscription(
            PoseWithCovarianceStamped, "/localization_pose", lambda _m: self._on_loc(), 10
        )
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.initialpose = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        self.scans: list[LaserScan] = []
        self.grid: OccupancyGrid | None = None
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.log: dict = {"steps": []}

    def _on_loc(self) -> None:
        self.localized_at = time.monotonic()

    def map_pose(self) -> tuple[float, float, float]:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                t = self.tf.lookup_transform("map", "base_link", Time())
                return (
                    t.transform.translation.x,
                    t.transform.translation.y,
                    math.degrees(quaternion_to_yaw(t.transform.rotation)),
                )
            except TransformException:
                if time.monotonic() > deadline:
                    raise RuntimeError("map -> base_link が引けない") from None
                self.spin_for(0.1)

    def call(self, name: str) -> None:
        client = self.create_client(Empty, name)
        try:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(f"{name} が見つからない")
            future = client.call_async(Empty.Request())
            deadline = time.monotonic() + 5.0
            while not future.done():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{name} が応答しない")
                self.spin_for(0.05)
        finally:
            self.destroy_client(client)

    def localize(self) -> None:
        self.call("/rtabmap/resume")
        since = time.monotonic()
        self.spin_for(self.wait_localization)
        turns = 0
        while self.localized_at < since:
            if turns >= self.search_turns:
                raise RuntimeError("回っても地図の上の位置を取り直せない")
            self.get_logger().info(f"位置を探すため {self.search_step:.0f} 度回る")
            self.turn(self.search_step)
            turns += 1
            self.spin_for(3.0)
        self.log["localize"] = {"turns": turns, "pose": self.map_pose()}

    def go(self, x: float, y: float, yaw_deg: float) -> None:
        if not self.nav.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Nav2 の navigate_to_pose が無い")
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation = yaw_to_quaternion(math.radians(yaw_deg))
        send = self.nav.send_goal_async(goal)
        deadline = time.monotonic() + 10.0
        while not send.done():
            if time.monotonic() > deadline:
                raise RuntimeError("Nav2 が目標を受け取らない")
            self.spin_for(0.05)
        handle = send.result()
        if not handle.accepted:
            raise RuntimeError("Nav2 が目標を断った")
        result = handle.get_result_async()
        deadline = time.monotonic() + self.goal_timeout
        while not result.done():
            if time.monotonic() > deadline:
                cancel = handle.cancel_goal_async()
                self.spin_for(2.0)
                del cancel
                raise RuntimeError("待機位置まで時間内に着かない")
            self.spin_for(0.1)
        if result.result().status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"待機位置へ行けなかった（Nav2 の状態 {result.result().status}）")
        self.log["entry_reached"] = self.map_pose()

    def approach(self, place: Place) -> None:
        self.call("/rtabmap/pause")
        for _ in range(self.max_steps):
            now = self.map_pose()
            step = next_step(now, place)
            self.log["steps"].append({"pose": [round(v, 3) for v in now], "step": step and step.__dict__})
            if step is None:
                return
            if step.kind == "turn":
                self.turn(step.amount)
            else:
                self.translate(step.amount, sideways=step.kind == "left", min_speed=0.01, hold_heading=True)
        raise RuntimeError(f"{self.max_steps} 手で着かない")

    def run(self) -> int:
        places = load_places(self.places_file)
        if self.adjust_name:
            return self.adjust(places)
        if self.save_marker:
            return self.register_marker(places)
        if self.align_only:
            place = places.get(self.align_only)
            if place is None or place.marker is None:
                raise RuntimeError(f"場所 {self.align_only!r} が無いか、マーカーの基準が無い")
            self.wait_inputs()
            self.call("/rtabmap/pause")
            self.align(place.marker)
            self.reanchor(place)
            return 0
        if self.save_name:
            pose = self.map_pose()
            place = Place(self.save_name, tuple(round(v, 3) for v in pose), tuple(self.entry), self.standoff)
            save_place(self.places_file, place)
            self.get_logger().info(f"登録した: {place}")
            return 0
        self.wait_inputs()
        t0 = time.monotonic()
        here = self.map_pose()
        for p in places.values():
            if near(here, p, self.near_radius):
                self.get_logger().info(f"{p.name} のそばにいるので、待機位置へ出る")
                self.approach(exit_of(p))
                self.log["left"] = p.name
                break
        if self.leave_only:
            return 0
        if self.place_name not in places:
            raise RuntimeError(f"場所 {self.place_name!r} が無い。登録済み: {sorted(places)}")
        place = places[self.place_name]
        self.localize()
        self.go(*place.entry)
        self.check_fit()
        self.approach(place)
        if place.marker is not None:
            self.align(place.marker)
            self.reanchor(place)
        final = self.map_pose()
        long_lat = [round(v, 3) for v in final]
        self.log.update(place=self.place_name, final=long_lat, seconds=round(time.monotonic() - t0, 1))
        self.get_logger().info(f"着いた: {self.place_name} {long_lat}（{self.log['seconds']} 秒）")
        return 0

    def adjust(self, places: dict[str, Place]) -> int:
        if self.adjust_name not in places:
            raise RuntimeError(f"場所 {self.adjust_name!r} が無い。登録済み: {sorted(places)}")
        place = places[self.adjust_name]
        fwd, left, turn = self.nudge
        if abs(fwd) > 0.10 or abs(left) > 0.10 or abs(turn) > 10.0:
            raise RuntimeError("nudge は前後と左右 0.10 m、向き 10 度まで")
        self.wait_inputs()
        before = self.map_pose()
        if not near(before, place, 0.05):
            raise RuntimeError(f"{place.name} に着いていない（今 {before}）。先に place:={place.name} で行く")
        self.call("/rtabmap/pause")
        if abs(turn) > 0.2:
            self.turn(turn)
        if abs(left) > 0.002:
            self.translate(left, sideways=True, min_speed=0.01, hold_heading=True)
        if abs(fwd) > 0.002:
            self.translate(fwd, min_speed=0.01, hold_heading=True)
        after = self.map_pose()
        new = replace(place, pose=tuple(round(v, 3) for v in after))
        if place.marker is not None:
            new = replace(new, marker=self.measure_ref(place.marker.marker_id))
        save_place(self.places_file, new)
        self.log.update(adjusted=place.name, before=before, after=after)
        self.get_logger().info(f"登録し直した: {new}")
        return 0

    def measure_ref(self, marker_id: int) -> MarkerRef:
        ref = self.eye.reference(marker_id)
        if ref is None:
            raise RuntimeError(f"マーカー {marker_id} か、そのまわりの壁が見えない")
        return ref

    def register_marker(self, places: dict[str, Place]) -> int:
        if self.save_marker not in places:
            raise RuntimeError(f"場所 {self.save_marker!r} が無い。登録済み: {sorted(places)}")
        place = places[self.save_marker]
        self.wait_inputs()
        here = self.map_pose()
        if not near(here, place, 0.05):
            raise RuntimeError(f"{place.name} に着いていない（今 {here}）。先に place:={place.name} で行く")
        new = replace(place, marker=self.measure_ref(self.marker_id))
        save_place(self.places_file, new)
        self.log.update(marker_saved=place.name, marker=new.marker.__dict__)
        self.get_logger().info(f"マーカーの基準を登録した: {new.marker}")
        return 0

    def align(self, ref: MarkerRef) -> None:
        self.log["align"] = []
        for _ in range(self.align_tries):
            got = self.eye.measure(ref.marker_id)
            if got is None:
                raise RuntimeError(f"マーカー {ref.marker_id} か、そのまわりの壁が見えない")
            center, wall_yaw = got
            dth, fwd, left = align_step((float(center[0]), float(center[1])), wall_yaw, ref)
            self.log["align"].append(
                {"turn_deg": round(math.degrees(dth), 2), "fwd": round(fwd, 4), "left": round(left, 4)}
            )
            # 台車の小さな動きは 2〜5 mm と 0.3° ほど行き過ぎるので、それより少し広く取る
            if abs(math.degrees(dth)) < 0.5 and math.hypot(fwd, left) < 0.006:
                self.get_logger().info(f"マーカーに合わせた（{len(self.log['align'])} 回目で許容の内）")
                return
            if abs(fwd) > MAX_ALIGN_M or abs(left) > MAX_ALIGN_M or abs(math.degrees(dth)) > MAX_ALIGN_DEG:
                raise RuntimeError(
                    f"マーカーで直す量が大きすぎる（回る {math.degrees(dth):.1f} 度、前 {fwd:.3f}、左 {left:.3f}）"
                )
            if abs(math.degrees(dth)) >= 0.3:
                self.turn(math.degrees(dth))
            if abs(left) > 0.002:
                self.translate(left, sideways=True, min_speed=0.01, hold_heading=True)
            if abs(fwd) > 0.002:
                self.translate(fwd, min_speed=0.01, hold_heading=True)
        last = self.log["align"][-1]
        if abs(last["turn_deg"]) < 1.0 and math.hypot(last["fwd"], last["left"]) < 0.015:
            self.get_logger().warn(
                f"{self.align_tries} 回で許容に入らなかったが、1.5 cm と 1° の内なので止める: {last}"
            )
            return
        raise RuntimeError(f"{self.align_tries} 回でマーカーに合わない")

    def _on_scan(self, msg: LaserScan) -> None:
        self.scans = [*self.scans[-19:], msg]

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.grid = msg

    def check_fit(self) -> None:
        """推定の位置に /scan を置き、点が地図の空きのマスに落ちる割合で、推定が合っているかを確かめる。"""
        self.scans = []
        deadline = time.monotonic() + 10.0
        while (len(self.scans) < 10 or self.grid is None) and time.monotonic() < deadline:
            self.spin_for(0.1)
        if self.grid is None or len(self.scans) < 10:
            raise RuntimeError("/scan か /map が届かないので、推定の位置を確かめられない")
        pts = []
        for sc in self.scans:
            r = np.asarray(sc.ranges, dtype=np.float64)
            a = sc.angle_min + np.arange(len(r)) * sc.angle_increment
            ok = np.isfinite(r) & (r < 3.0)
            pts.append(np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], 1))
        g = self.grid.info
        cells = np.asarray(self.grid.data, dtype=np.int16).reshape(g.height, g.width)
        x, y, yaw_deg = self.map_pose()
        ratio, center_free = free_hit_ratio(
            np.concatenate(pts),
            cells,
            g.resolution,
            (g.origin.position.x, g.origin.position.y),
            (x, y, math.radians(yaw_deg)),
        )
        self.log["fit"] = {"free_hit_ratio": round(ratio, 3), "center_free": center_free}
        self.get_logger().info(f"推定の位置の確かめ: {self.log['fit']}")
        if ratio > MAX_FREE_HIT or not center_free:
            raise RuntimeError(f"推定の位置が地図と合わない（{self.log['fit']}）。寄せずに止める")

    def reanchor(self, place: Place) -> None:
        """マーカーで合わせ込んだあと、登録した位置を地図の上の位置として rtabmap に渡す。"""
        before = self.map_pose()
        x, y, yaw_deg = place.pose
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation = yaw_to_quaternion(math.radians(yaw_deg))
        self.call("/rtabmap/resume")
        self.initialpose.publish(msg)
        self.spin_for(1.5)
        self.call("/rtabmap/pause")
        after = self.map_pose()
        self.log["reanchor"] = {
            "before": [round(v, 3) for v in before],
            "after": [round(v, 3) for v in after],
        }
        self.get_logger().info(f"地図の上の位置を決め直した: {self.log['reanchor']}")

    def write(self) -> None:
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        with open(self.out, "w") as f:
            json.dump(self.log, f, indent=1, ensure_ascii=False)


def main() -> None:
    rclpy.init()
    node = GotoPlace()
    code = 1
    try:
        code = node.run()
    except KeyboardInterrupt:
        code = 130
    except (RuntimeError, ValueError) as exc:
        node.log["error"] = str(exc)
        node.get_logger().error(str(exc))
    finally:
        node.hold(0.5)
        node.write()
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(code)
