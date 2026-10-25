"""iPad の深度で天板の手前の縁を測り、台車を縁に正対させて、決めた隙間まで寄せる。

    ros2 run xlerobot_nav table_dock --ros-args -p gap:=0.04 -p strafe:=0.15 -p out:=/out/dock/run.json

nav.launch.py か map.launch.py を起動し、Nav2 に目標を送っていない状態で使う。
机の手前（レンズから縁まで 0.6〜1.0 m）で、机の方をおおよそ向いてから始める。

1. 止まって深度を frames 枚重ね、縁の線を当てはめる
2. 縁に正対するまで回り、測り直す
3. strafe [m] だけ横へ動き、測り直す。横は深度で見えないので、空きは使う人が確かめておく
4. レンズから縁までが last_look [m] になるまで進み、最後にもう一度測る。縁が視野の上へ外れる手前で測るため
5. 縁はもう見えないので、融合した odom で残りを進み、台車の前の縁と天板の縁のあいだを gap にして止まる
6. hold 秒止まり、トラッキングの状態、ARKit の姿勢、融合した odom、地図の上の位置を記録する
7. 4 で測った位置（さらに backoff [m]）まで下がって、縁を測り直す。stay なら下がらずに終わる

止まっていたときの隙間は、3 通りに求める。
- odom: 最後に測った縁の距離から、odom で進んだ距離を引く
- 奥の壁: 縁の下の奥に見える壁は、寄せたあとも見え続ける。最後に測ったときの壁と縁の差を、寄せたときの壁の距離から引く
- 下がって測り直す: 最後に測ったときと同じ距離から測った縁の距離から、odom で下がった距離を引く

結果は out に JSON で書く。dry_run なら 1 の測定だけをして終わる。

rtabmap が動いていれば（map.launch.py）、1 の測定のあとで位置の取り直しを止め、7 の測り直しのあとで再開する。
stay なら止めたまま終わる。机の前に留まるあいだも位置が跳ばないように、離れるときに /rtabmap/resume を呼ぶ。
途中で止まったときも再開する。机に近い視野で取り直すと、地図の上の位置が跳ぶことがあるため。
止めているあいだの地図の上の位置は、止めたときの map -> odom と融合した odom で決まる。
rtabmap が動いていなければ（nav.launch.py）、止めずに寄せる。pause_service を空にすると、動いていても止めない。

安全の約束:
- 回る角が max_turn_deg を、進む距離が max_approach を超える計画は、動かさずに終わる
- 最後の前進は、縁までの距離から min_gap を残す。台車の通り道（台車の幅に左右 3 cm を足した範囲）に
  縁より手前の物（机の脚や柱）が見えていれば、その物から min_gap を残す
- 最後の測定の縁が、直前の測定と odom の移動量から見立てた位置と 3 cm 以上違えば、寄せずに終わる
- 動きの共通の約束（時間の上限、odom の途切れ）は odom_driver にある
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import warnings

import numpy as np
import rclpy
from navgate.table_edge import (
    Camera,
    Edge,
    approach_distance,
    docked_gap,
    find_edge,
    min_visible_distance,
    nearest_in_corridor,
    points_from_depth,
    wall_ahead,
)
from pocketsensor_msgs.msg import TrackingStatus
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from xlerobot_nav.msgs import transform_to_matrix
from xlerobot_nav.odom_driver import OdomDriver

DEPTH = "/pocketsensor/depth"


def _pose2d(T: np.ndarray) -> list[float]:
    return [float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0])]


def _edge_dict(edge: Edge | None) -> dict | None:
    if edge is None:
        return None
    return dataclasses.asdict(edge) | {"yaw_deg": math.degrees(edge.yaw), "distance": edge.distance}


def line_in(pose: list[float], edge: Edge) -> dict:
    """base_link で見た縁の線を、pose（x, y, yaw）の座標で表す。法線（天板の奥へ）の向きと、原点から線までの距離。"""
    x, y, th = pose
    phi = th + edge.yaw
    fx, fy = x + edge.distance * math.cos(phi), y + edge.distance * math.sin(phi)
    return {"normal_deg": math.degrees(phi), "c": fx * math.cos(phi) + fy * math.sin(phi), "foot": [fx, fy]}


class TableDock(OdomDriver):
    def __init__(self) -> None:
        super().__init__("table_dock")

        def param(name: str, default):
            return self.declare_parameter(name, default).value

        self.gap = float(param("gap", 0.04))
        self.min_gap = float(param("min_gap", 0.015))
        # 台車の前の縁（base_link の x）。Nav2 の footprint と同じ値
        self.front_x = float(param("front_x", 0.175))
        self.strafe = float(param("strafe", 0.0))
        self.last_look = float(param("last_look", 0.58))
        self.hold_s = float(param("hold", 20.0))
        self.backoff = float(param("backoff", 0.0))
        self.stay = bool(param("stay", False))
        self.frames = int(param("frames", 15))
        self.speed = float(param("speed", 0.05))
        self.final_speed = float(param("final_speed", 0.03))
        self.turn_speed = float(param("turn_speed", 10.0))
        self.align_tol = float(param("align_tol_deg", 0.5))
        self.max_turn = float(param("max_turn_deg", 25.0))
        self.max_approach = float(param("max_approach", 0.9))
        # 台車の通り道の半分の幅 [m]。footprint の半分の幅 0.225 に 3 cm を足す
        self.corridor = float(param("corridor_half_width", 0.255))
        self.out = str(param("out", "/out/dock/table_dock.json"))
        self.dry_run = bool(param("dry_run", False))
        # rtabmap の位置の取り直しを止める、再開するサービス。空なら止めない
        self.pause_service = str(param("pause_service", "/rtabmap/pause"))
        self.resume_service = str(param("resume_service", "/rtabmap/resume"))
        self.K: tuple[float, float, float, float] | None = None
        self.size = (0, 0)
        self.depth_frame = ""
        self.want = 0
        self.buf: list[np.ndarray] = []
        self.tracking: list[int] | None = None
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(Image, f"{DEPTH}/image", self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f"{DEPTH}/camera_info", self._on_info, qos_profile_sensor_data)
        self.create_subscription(TrackingStatus, "/pocketsensor/tracking", self._on_tracking, 20)
        names = (
            "gap",
            "min_gap",
            "front_x",
            "strafe",
            "last_look",
            "hold_s",
            "backoff",
            "stay",
            "frames",
            "speed",
            "final_speed",
            "pause_service",
            "resume_service",
        )
        self.result: dict = {"params": {k: getattr(self, k) for k in names}, "measurements": [], "moves": []}

    def _on_info(self, msg: CameraInfo) -> None:
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        self.size = (msg.width, msg.height)
        self.depth_frame = msg.header.frame_id

    def _on_tracking(self, msg: TrackingStatus) -> None:
        self.tracking = [int(msg.state), int(msg.reason), int(msg.origin_epoch)]

    def _on_depth(self, msg: Image) -> None:
        if self.want <= 0:
            return
        depth = (
            np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
        )
        depth[depth == 0] = np.nan
        self.buf.append(depth / 1000.0)
        self.want -= 1

    def _lookup(self, target: str, source: str) -> np.ndarray | None:
        try:
            return transform_to_matrix(self.tf.lookup_transform(target, source, Time()))
        except TransformException:
            return None

    def _odom(self) -> list[float]:
        odom = self.tracks["odom"]
        x, y = odom.xy()
        return [x, y, odom.yaw.value]

    def _map(self) -> list[float] | None:
        T = self._lookup("map", "base_link")
        return _pose2d(T) if T is not None else None

    def camera(self) -> Camera:
        T = self._lookup("base_link", self.depth_frame)
        if T is None or self.K is None:
            raise RuntimeError("深度の camera_info か、base_link からの TF が無い")
        return Camera(K=self.K, width=self.size[0], height=self.size[1], T_base_opt=T)

    def measure(self, label: str, expect_offset: float | None = None) -> Edge | None:
        self.hold(0.6)  # 止まってから、動いていたあいだの深度が流れ切るまで待つ
        self.buf, self.want = [], self.frames
        deadline = time.monotonic() + 5.0
        while self.want > 0:
            if time.monotonic() > deadline:
                raise RuntimeError("深度が届かない")
            self.send()
            self.spin_for(0.05)
        cam = self.camera()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # どのフレームでも無効な画素は NaN のまま
            depth = np.nanmedian(np.stack(self.buf), axis=0)
        points = points_from_depth(depth, cam.K, cam.T_base_opt)
        edge = find_edge(points, camera=cam)
        # 縁の下の奥の壁。縁が見えなければ、見立てた縁の位置より奥を探す
        # 見立てた縁の位置があればそれを使う。寄せたあとに見える別の物（横の壁など）を縁と取り違えないため
        near = expect_offset if expect_offset is not None else (edge.offset if edge is not None else None)
        lens_y = float(np.asarray(cam.T_base_opt)[1, 3])
        wall = wall_ahead(points, lens_y=lens_y, beyond=near + 0.10) if near is not None else None
        corridor = nearest_in_corridor(points, half_width=self.corridor)
        odom, mp = self._odom(), self._map()
        rec = {
            "label": label,
            "t": time.time(),
            "edge": _edge_dict(edge),
            "wall": wall,
            "corridor_nearest": corridor,
            "odom": odom,
            "map": mp,
            "arkit": self.tracks["arkit"].snapshot(),
            "tracking": self.tracking,
        }
        if edge is not None:
            rec["edge_in_odom"] = line_in(odom, edge)
            if mp is not None:
                rec["edge_in_map"] = line_in(mp, edge)
            self.get_logger().info(
                f"{label}: distance {edge.distance:.3f} m, yaw {math.degrees(edge.yaw):+.1f} deg, "
                f"y [{edge.y_right:+.2f}, {edge.y_left:+.2f}], height {edge.height:.2f}, bins {edge.bins}, "
                f"below {edge.support_below:.2f}, wall {wall if wall is None else round(wall, 3)}"
            )
        else:
            self.get_logger().warning(
                f"{label}: 縁が見つからない。奥の壁 {wall if wall is None else round(wall, 3)}"
            )
        self.result["measurements"].append(rec)
        return edge

    def move(self, kind: str, amount: float, **kwargs) -> float:
        start, t0 = self._odom(), time.monotonic()
        if kind == "turn":
            self.turn(amount)
            moved = math.degrees(self._odom()[2] - start[2])
        else:
            moved = self.translate(amount, sideways=kind == "left", **kwargs)
        self.result["moves"].append(
            {
                "kind": kind,
                "amount": amount,
                "moved": moved,
                "start": start,
                "end": self._odom(),
                "seconds": time.monotonic() - t0,
            }
        )
        return moved

    def square_up(self, edge: Edge, label: str) -> Edge:
        deg = math.degrees(edge.yaw)
        if abs(deg) <= self.align_tol:
            return edge
        if abs(deg) > self.max_turn:
            raise RuntimeError(f"縁に正対するには {deg:+.1f}° 回る必要があり、上限 {self.max_turn}° を超える")
        self.move("turn", deg)
        again = self.measure(label)
        if again is None:
            raise RuntimeError(f"{label}: 回ったあと、縁が見つからない")
        return again

    def record_hold(self, seconds: float) -> dict:
        samples, end, next_t = [], time.monotonic() + seconds, 0.0
        while time.monotonic() < end:
            self.send()
            self.spin_for(0.05)
            if time.monotonic() >= next_t:
                next_t = time.monotonic() + 0.2
                arkit = self.tracks["arkit"]
                samples.append(
                    {
                        "t": time.time(),
                        "tracking": self.tracking,
                        "odom": self._odom(),
                        "arkit": [*arkit.xy(), arkit.yaw.value],
                        "map": self._map(),
                    }
                )

        def drift(key: str) -> list[float] | None:
            rows = np.array([s[key] for s in samples if s[key] is not None])
            if len(rows) < 2:
                return None
            d = rows - rows[0]
            return [float(np.hypot(d[:, 0], d[:, 1]).max()), float(np.degrees(np.abs(d[:, 2]).max()))]

        states = [s["tracking"][0] for s in samples if s["tracking"] is not None]
        epochs = {s["tracking"][2] for s in samples if s["tracking"] is not None}
        return {
            "seconds": seconds,
            "normal_fraction": states.count(2) / len(states) if states else None,
            "reasons": sorted({s["tracking"][1] for s in samples if s["tracking"] is not None}),
            "origin_epochs": sorted(epochs),
            # 動いていないはずの台車で、位置 [m] と yaw [deg] がどれだけ動いて見えたか
            "drift": {k: drift(k) for k in ("odom", "arkit", "map")},
            "samples": samples,
        }

    def _call(self, name: str, timeout: float = 3.0, wait: float | None = None) -> None:
        client = self.create_client(Empty, name)
        try:
            if not client.wait_for_service(timeout_sec=timeout if wait is None else wait):
                raise LookupError(f"{name} が見つからない")
            future = client.call_async(Empty.Request())
            deadline = time.monotonic() + timeout
            while not future.done():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{name} が応答しない")
                self.spin_for(0.05)
        finally:
            self.destroy_client(client)

    def _resume(self) -> None:
        # 元の例外を隠さないよう、再開の失敗はここで記録して終える
        try:
            self._call(self.resume_service)
            self.result["localization"]["resumed"] = time.time()
        except (LookupError, RuntimeError) as exc:
            self.result["localization"]["resume_error"] = str(exc)
            self.get_logger().error(f"位置の取り直しを再開できない: {exc}")

    def run(self) -> None:
        self.wait_inputs()
        deadline = time.monotonic() + 10.0
        while self.K is None:
            if time.monotonic() > deadline:
                raise RuntimeError("深度の camera_info が届かない")
            self.spin_for(0.1)
        edge = self.measure("start")
        if edge is None:
            raise RuntimeError("縁が見つからない。机の方を向いているか、近すぎないかを確かめる")
        if edge.support_below > 0.5:
            self.get_logger().warning("縁の真下にも点がある。天板ではなく、壁や箱かもしれない")
        if self.dry_run:
            return
        paused = False
        if self.pause_service:
            try:
                self._call(self.pause_service, wait=1.0)
                paused = True
            except LookupError:
                self.get_logger().info("rtabmap が動いていないので、位置の取り直しは止めずに寄せる")
        if not paused:
            self._dock(edge)
            return
        self.result["localization"] = {"paused": time.time()}
        try:
            self._dock(edge)
        except BaseException:
            self._resume()
            raise
        if self.stay and "results" in self.result:
            self.get_logger().info(
                "机の前に留まる。位置の取り直しは止めたまま。離れるときに /rtabmap/resume を呼ぶ"
            )
        else:
            self._resume()

    def _dock(self, edge: Edge) -> None:
        edge = self.square_up(edge, "squared")
        if abs(self.strafe) > 0.001:
            self.move("left", self.strafe, hold_heading=True)
            again = self.measure("strafed")
            if again is None:
                raise RuntimeError("横へ動いたあと、縁が見つからない")
            edge = self.square_up(again, "squared_after_strafe")

        cam = self.camera()
        lens_x = float(np.asarray(cam.T_base_opt)[0, 3])
        # 縁が視野の上へ外れる距離より手前で、最後に測る
        last_look = max(self.last_look, min_visible_distance(cam, edge.height + 0.015) + 0.02)
        self.result["last_look_used"] = last_look
        to_last = (edge.distance - lens_x) - last_look
        if to_last > 0.01:
            if to_last > self.max_approach:
                raise RuntimeError(f"最後に測る位置まで {to_last:.2f} m あり、上限を超える")
            moved = self.move("fwd", to_last, hold_heading=True)
            again = self.measure("last_look")
            if again is None:
                raise RuntimeError("最後の測定で縁が見つからない")
            expected = edge.distance - moved
            if abs(again.distance - expected) > 0.03:
                raise RuntimeError(
                    f"縁の見立てが odom の移動量と {again.distance - expected:+.3f} m 合わない"
                )
            edge = self.square_up(again, "last_look_squared") if abs(math.degrees(again.yaw)) > 1.5 else again

        last = self.result["measurements"][-1]
        wall_last, corridor = last["wall"], last["corridor_nearest"]
        limits = [
            approach_distance(edge, front_x=self.front_x, gap=self.gap),
            edge.distance - self.front_x - self.min_gap,
        ]
        if corridor is not None:
            limits.append(corridor - self.front_x - self.min_gap)
            if corridor < edge.distance - 0.05:
                self.get_logger().warning(
                    f"通り道の {corridor:.3f} m に、縁より手前の物がある。その手前で止める"
                )
        forward = min(limits)
        if forward > self.max_approach:
            raise RuntimeError(f"寄せるのに {forward:.2f} m 進む必要があり、上限を超える")
        # 遅い前進では、車輪の odom が実際の進みより 1〜4 % 短く出た。奥の壁が見えていれば、
        # 残り 6 cm の手前でいったん止まり、壁で測った進みから残りを決める
        odom_moved, est_moved, approach_by = 0.0, 0.0, "odom"
        self.speed, saved = self.final_speed, self.speed
        try:
            if wall_last is not None and forward > 0.08:
                first = self.move("fwd", forward - 0.06, min_speed=0.01, hold_heading=True)
                self.measure("near_stop", expect_offset=edge.offset - first)
                wall_now = self.result["measurements"][-1]["wall"]
                progress = first
                if wall_now is not None:
                    progress = wall_last - wall_now
                    if abs(progress - first) > 0.05:
                        raise RuntimeError(
                            f"奥の壁で測った進み {progress:.3f} m が、odom の {first:.3f} m と合わない"
                        )
                    approach_by = "wall"
                rest = forward - progress
                second = (
                    self.move("fwd", rest, min_speed=0.01, hold_heading=True) if abs(rest) > 0.003 else 0.0
                )
                odom_moved, est_moved = first + second, progress + second
            elif forward > 0.005:
                odom_moved = est_moved = self.move("fwd", forward, min_speed=0.01, hold_heading=True)
        finally:
            self.speed = saved
        results: dict = {
            "gap_target": self.gap,
            "approach_by": approach_by,
            "gap_from_odom": edge.distance - self.front_x - odom_moved,
            "gap_planned": edge.distance - self.front_x - est_moved,
        }
        seen = self.measure("docked", expect_offset=edge.offset - est_moved)
        if seen is not None and abs(seen.offset - (edge.offset - est_moved)) < 0.10:
            self.get_logger().warning(
                "寄せたあとも縁が見えている。縁の見立てと違うものを見ているかもしれない"
            )
        wall_docked = self.result["measurements"][-1]["wall"]
        if wall_last is not None and wall_docked is not None:
            results["gap_from_wall"] = wall_docked - (wall_last - edge.distance) - self.front_x
        self.get_logger().info(
            f"寄せた（{approach_by}）: 隙間は odom から {results['gap_from_odom'] * 100:.1f} cm、"
            f"奥の壁から {results.get('gap_from_wall', math.nan) * 100:.1f} cm。{self.hold_s:.0f} 秒止まる"
        )
        self.result["hold"] = self.record_hold(self.hold_s)
        results["yaw_last_deg"] = math.degrees(edge.yaw)
        if self.stay:
            self.result["results"] = results
            self.get_logger().info(f"結果: {json.dumps(results)}")
            return

        back = self.move("fwd", -(est_moved + self.backoff), hold_heading=True)
        after = self.measure("after_backoff")
        if after is not None:
            results["gap_from_backoff"] = docked_gap(
                distance_after=after.distance, backoff=-back, front_x=self.front_x
            )
            results["yaw_after_deg"] = math.degrees(after.yaw)
        results["yaw_last_deg"] = math.degrees(edge.yaw)
        self.result["results"] = results
        self.get_logger().info(f"結果: {json.dumps(results)}")

    def write(self) -> None:
        os.makedirs(os.path.dirname(self.out) or ".", exist_ok=True)
        with open(self.out, "w") as f:
            json.dump(self.result, f, indent=1)
        self.get_logger().info(f"wrote {self.out}")


def main() -> None:
    rclpy.init()
    node = TableDock()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        node.result["error"] = str(exc)
        node.get_logger().error(str(exc))
    finally:
        node.hold(0.5)
        node.write()
        node.destroy_node()
        rclpy.try_shutdown()
