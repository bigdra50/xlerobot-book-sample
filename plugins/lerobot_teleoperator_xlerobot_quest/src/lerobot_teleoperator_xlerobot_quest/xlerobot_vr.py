"""Quest の VR 入力を XLeRobot の 17 キー action にする Teleoperator。

    Quest（WebXR） --wss--> teleop_xr.Teleop（このプロセスのスレッド）
        --XRState--> IK（既定は閉形式。ik_backend で切り替え） --関節角(rad)-->
        可動域 clamp → スルーレート制限 → 正規化値 → get_action()

teleop_xr が担うのは VR 入力の取得と、ik_backend=pyroki のときの数値 IK。
engage（いつ追従するか）、安全機構、単位変換はここで持つ。
closed_form は腕ごとに解析解を出し、そのあとは同じ経路を通る。

lerobot-teleoperate / lerobot-record からは --teleop.type=xlerobot_quest で使う。
teleop_xr と jax は connect() で初めて import する。
"""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .base import stick_to_base_velocity
from .config_xlerobot_vr import IKBackend, XLeRobotVRConfig
from .contracts import (
    ACTION_KEYS,
    BASE_KEYS,
    HEAD_KEYS,
    IK_JOINTS,
    MOTOR_NAMES,
    SIDES,
    EngagePhase,
    JointLimits,
    VRSnapshot,
    arm_motor,
    ik_joint_name,
)
from .engage import EngageStateMachine
from .gripper import trigger_to_gripper
from .head import HeadFollower
from .record_control import RecordControl
from .rest import RestReturn, load_rest
from .safety import SlewLimiter, StaleGate, clamp_to_limits, max_step_from_speed
from .units import from_robot_units, limits_from_calibration, to_robot_units
from .xr_state import snapshot_from_xr_state

logger = logging.getLogger(__name__)

HEAD_PAN, HEAD_TILT = (k.removesuffix(".pos") for k in HEAD_KEYS)
# 度で上下限を持つモーター。gripper は 0..100 の % なので度の上下限を当てない
ANGLE_MOTORS: tuple[str, ...] = tuple(m for m in MOTOR_NAMES if not m.endswith("gripper"))
GRIPPER_MOTORS: tuple[str, ...] = tuple(m for m in MOTOR_NAMES if m.endswith("gripper"))


class XLeRobotVR(Teleoperator):
    config_class = XLeRobotVRConfig
    name = "xlerobot_quest"

    def __init__(self, config: XLeRobotVRConfig):
        # 親クラスが calibration_dir/<id>.json を self.calibration へ読む。
        # 公式 17 台 1 ファイル形式のキー（left_arm_shoulder_pan など）がそのまま使える
        super().__init__(config)
        self.config = config
        self._limits = {
            m: limits_from_calibration(self.calibration[m]) for m in ANGLE_MOTORS if m in self.calibration
        }
        lo, hi = sorted((config.gripper_open, config.gripper_closed))
        self._gripper_limits = {m: JointLimits(lower_deg=lo, upper_deg=hi) for m in GRIPPER_MOTORS}
        self._engage = EngageStateMachine()
        # head_motor_1 は正が右回り（head カメラの画像で確かめた）。HMD の yaw は左回りが正なので逆にする。
        # head_motor_2 は正が下向きで、HMD の pitch も下向きが正なのでそのまま
        self._head = HeadFollower(pan_sign=-1.0)
        step = max_step_from_speed(config.max_joint_speed_dps, config.fps)
        # gripper は % だが、1 ティックあたりの上限は度と同じ値で縛る。別に持つほどの根拠が無い
        self._slew = SlewLimiter(dict.fromkeys(MOTOR_NAMES, step))
        self._stale = StaleGate(config.stale_timeout_s)

        self._lock = threading.Lock()
        self._latest_state: Any = None  # teleop_xr.messages.XRState
        self._latest_snap: VRSnapshot | None = None
        self._q: np.ndarray | None = None  # IK の関節角（rad）。robot.actuated_joint_names の順
        self._joint_index: dict[str, int] = {}
        self._hold_deg: dict[str, float] | None = None  # 直近に出した指令（モーター度。gripper は %）
        self._warned_no_reference = False
        self._warned_bad_message = False
        self._last_phase = self._engage.phase

        self._robot: Any = None
        self._controller: Any = None
        self._server: Any = None
        self._uvicorn: Any = None
        self._server_thread: threading.Thread | None = None
        self._xr_state_cls: Any = None
        self._head_video: Any = None  # teleop_xr の ExternalVideoSource
        self._last_head_frame: np.ndarray | None = None
        self._reset_started = threading.Event()
        self._correction_started = threading.Event()
        self._record = RecordControl(
            on_reset=self._reset_started.set, on_correction_start=self._correction_started.set
        )
        self._frames_stop = threading.Event()
        self._frames_thread: threading.Thread | None = None
        self._rest: RestReturn | None = None
        self._prev_b = self._prev_y = self._prev_x = False

    # --- Teleoperator の契約 ------------------------------------------------

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server_thread is not None and self._server_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        # モーターを持たないので、キャリブレーションの対象が無い
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        try:
            import teleop_xr
            from teleop_xr import Teleop
            from teleop_xr.config import TeleopSettings
            from teleop_xr.messages import XRState
        except ImportError as e:
            raise ImportError(
                "teleop_xr が無い。pip install 'lerobot_teleoperator_xlerobot_quest[vr]' で入る"
            ) from e
        # 可動域の clamp と正規化値への変換は、どちらもキャリブレーションの範囲から作る
        if not self.calibration:
            raise RuntimeError(
                "robot のキャリブレーション JSON が要る。"
                "--teleop.calibration_dir と --teleop.id で公式 17 台 1 ファイル形式を指す"
            )

        if self.config.rest_path:
            rest_deg = load_rest(Path(self.config.rest_path))
            self._rest = RestReturn(rest_deg, self._limits, self.config.gripper_closed)

        if self.config.ik_backend is IKBackend.PYROKI:
            self._connect_pyroki()
        else:
            self._connect_analytic()
        self._xr_state_cls = XRState

        # Quest 内に腕のモデルは出さない（robot_vis を渡さない）。関節角を送らないので、出しても動かない
        camera_views, video_sources = self._head_camera()
        settings = TeleopSettings(host=self.config.host, port=self.config.port, camera_views=camera_views)
        self._server = Teleop(settings=settings, video_sources=video_sources)
        self._server.set_pose(np.eye(4))
        self._server.subscribe(self._on_xr_update)
        self._server_thread = threading.Thread(
            target=self._serve,
            args=(Path(teleop_xr.__file__).parent,),
            daemon=True,
            name="xlerobot-vr-server",
        )
        self._server_thread.start()
        # dagger では、ポリシーが動いているあいだ get_action が呼ばれない。映像は get_action と別に送り続ける
        self._frames_stop.clear()
        self._frames_thread = threading.Thread(
            target=self._push_frames, daemon=True, name="xlerobot-vr-frames"
        )
        self._frames_thread.start()
        self._record.install()
        logger.info(
            "Quest のブラウザで https://%s:%d を開き Enter VR を押す（自己署名証明書の警告は詳細から続行）",
            teleop_xr.get_local_ip(),
            self.config.port,
        )

    def _connect_pyroki(self) -> None:
        """同梱の両腕 URDF を 1 つの PyrokiSolver で解く。"""
        try:
            from teleop_xr.ik.solver import PyrokiSolver

            from .ik.so101_dual import SO101DualArmRobot
        except ImportError as e:
            raise ImportError(
                "PyRoki が無い。pip install 'lerobot_teleoperator_xlerobot_quest[vr]' で入る。"
                "jax を使わないなら --teleop.ik_backend=closed_form"
            ) from e

        self._robot = SO101DualArmRobot(joint_limits_rad=self._ik_limits_rad())
        self._joint_index = {name: i for i, name in enumerate(self._robot.actuated_joint_names)}
        logger.info("IK backend=pyroki。JIT warmup を始める（初回は数十秒）")
        solver = PyrokiSolver(self._robot)
        if not solver.warmup_complete:
            raise RuntimeError(f"PyRoki の warmup に失敗: {solver.warmup_error}")
        self._controller = _engage_ik_controller(self._robot, solver)
        self._q = np.asarray(self._robot.get_default_config(), dtype=float)

    def _connect_analytic(self) -> None:
        """腕ごとに解析 IK を解く。URDF も PyRoki も作らない。"""
        from .ik.analytic import ANALYTIC_JOINTS, AnalyticIKController, ClosedFormIK

        self._controller = AnalyticIKController(ClosedFormIK())
        self._joint_index = {name: i for i, name in enumerate(ANALYTIC_JOINTS)}
        self._q = np.zeros(len(ANALYTIC_JOINTS))
        logger.info("IK backend=%s。左右の腕を別々に解く", self.config.ik_backend.value)

    def _head_camera(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Quest の正面に head カメラを出す設定。フレームは get_action で観測から渡す。"""
        from teleop_xr.config import ViewConfig
        from teleop_xr.video_stream import ExternalVideoSource

        # teleop_xr にカメラを開かせない（device="external"）。put_frame で渡した画像だけを出す
        self._head_video = ExternalVideoSource()
        view = ViewConfig(device="external", width=1280, height=720, fps=30)
        return {"head": view}, {"head": self._head_video}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        assert self._server_thread is not None
        self._server_thread.join(timeout=5.0)
        self._server_thread = None
        self._frames_stop.set()
        if self._frames_thread is not None:
            self._frames_thread.join(timeout=1.0)
            self._frames_thread = None
        self._engage.reset()
        self._record.uninstall()

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        now = time.monotonic()
        with self._lock:
            state, snap = self._latest_state, self._latest_snap

        if self._correction_started.is_set():
            self._correction_started.clear()
            # dagger では補正のあいだしか get_action が呼ばれない。直近の指令は前回の補正の終わりの姿勢なので、
            # そこへ跳ばないよう基準をロボットの今の姿勢から取り直す。追従は A を押し直してから始める
            self._hold_deg = None
            self._engage.reset()
            self._reset_started.clear()

        if self._hold_deg is None:
            reference = self._observed_pose_deg()
            if reference is None:
                if not self._warned_no_reference:
                    logger.warning(
                        "ロボットの現在姿勢が分からないので腕と head の指令を出さない。"
                        "--robot.type=xlerobot_book_client と一緒に使う"
                    )
                    self._warned_no_reference = True
                return dict.fromkeys(BASE_KEYS, 0.0)
            self._hold_deg = reference
            self._q = self._q_from_motor_deg(reference)
            logger.info("基準姿勢を取った: %s", self._fmt_deg(reference))

        target = dict(self._hold_deg)
        base = dict.fromkeys(BASE_KEYS, 0.0)

        if snap is None or not self._stale.is_fresh(now):
            # 入力が途絶えたら追従を切る。再開には A を押し直す。自走させない
            if self._engage.phase is not EngagePhase.IDLE:
                logger.info("追従を切った: VR の入力が %.2f 秒届かなかった", self._stale.age(now))
                self._engage.reset()
        else:
            was_following = self._engage.phase is not EngagePhase.IDLE
            event = self._engage.update(snap)
            if was_following and event.phase is EngagePhase.IDLE:
                logger.info("追従を切った: %s", idle_reason(snap))
            if event.rebase:
                self._resync_from_robot()
                target = dict(self._hold_deg)  # type: ignore[arg-type]
                self._controller.reset()
                self._head.rebase(snap.head, target[HEAD_PAN], target[HEAD_TILT])
            if event.engaged:
                q_new = np.asarray(self._controller.step(state, self._q), dtype=float)
                for side in SIDES:
                    for joint in IK_JOINTS:
                        motor = arm_motor(side, joint)
                        target[motor] = math.degrees(
                            float(q_new[self._joint_index[ik_joint_name(side, joint)]])
                        )
                target[arm_motor("left", "gripper")] = self._gripper_value(snap.left.trigger)
                target[arm_motor("right", "gripper")] = self._gripper_value(snap.right.trigger)
                if self.config.follow_head:
                    target[HEAD_PAN], target[HEAD_TILT] = self._head.update(snap.head)
                base = stick_to_base_velocity(
                    snap.left,
                    snap.right,
                    xy_speed_mps=self.config.base_xy_speed_mps,
                    theta_speed_dps=self.config.base_theta_speed_dps,
                    deadzone=self.config.stick_deadzone,
                )

        if self._reset_started.is_set():
            self._reset_started.clear()
            if self._rest is not None:
                self._engage.reset()
                self._rest.start()
        if self._rest is not None and self._rest.active:
            # A で追従を始めたら、戻る途中でも人の操作を優先する
            if self._engage.phase is EngagePhase.ENGAGED:
                self._rest.cancel()
            else:
                target = self._rest.step(target, self._observed_pose_deg(), now)

        target = clamp_to_limits(target, self._limits)
        target = clamp_to_limits(target, self._gripper_limits)
        target = self._slew.step(self._hold_deg, target)
        self._hold_deg = target
        self._log_phase()
        # 制限後の指令を IK の初期値に戻す。IK の解と実際に出した指令を乖離させない
        self._q = self._q_from_motor_deg(target)

        normalized = to_robot_units(target, self.calibration)
        action: dict[str, float] = {f"{m}.pos": float(normalized[m]) for m in MOTOR_NAMES}
        action.update(base)
        return action

    # --- 内側 ----------------------------------------------------------------

    @staticmethod
    def _fmt_deg(values: dict[str, float]) -> str:
        return " ".join(f"{m.replace('_arm_', ':')}={v:.1f}" for m, v in values.items())

    def _log_phase(self) -> None:
        phase = self._engage.phase
        if phase != self._last_phase:
            logger.info("engage: %s -> %s", self._last_phase.value, phase.value)
            self._last_phase = phase

    def _serve(self, package_dir: Path) -> None:
        import uvicorn

        # teleop_xr.Teleop.run() は uvicorn.run() を呼ぶだけで止める手段が無い。
        # Server を自分で持てば should_exit で止められる。証明書は teleop_xr 同梱のもの
        config = uvicorn.Config(
            self._server.app,
            host=self.config.host,
            port=self.config.port,
            ssl_keyfile=str(package_dir / "key.pem"),
            ssl_certfile=str(package_dir / "cert.pem"),
            log_level="warning",
        )
        self._uvicorn = uvicorn.Server(config)
        self._uvicorn.run()

    def _on_xr_update(self, _pose: np.ndarray, message: dict[str, Any]) -> None:
        # teleop_xr の購読は uvicorn のスレッドから呼ばれる。ここでは受け取って置くだけ
        data = message.get("data", message)
        try:
            state = self._xr_state_cls.model_validate(data)
        except Exception:
            # teleop_xr とメッセージの形が合わないと、入力が届かないまま止まる。原因を追えるよう初回だけ出す
            if not self._warned_bad_message:
                logger.warning("XRState として読めないメッセージを捨てた", exc_info=True)
                self._warned_bad_message = True
            return
        t = time.monotonic()
        # mode="json" で Enum を文字列にする。role の比較を Enum 型に依存させない
        snap = snapshot_from_xr_state(state.model_dump(mode="json"))
        with self._lock:
            first = self._latest_snap is None
            self._latest_state = state
            self._latest_snap = snap
        self._stale.note_input(t)
        # get_action が呼ばれない記録の開始前もボタンを拾うよう、届いた入力ごとに見る
        if snap.right.secondary and not self._prev_b:
            self._record.press("right")
        if snap.left.secondary and not self._prev_y:
            self._record.press("left")
        if snap.left.primary and not self._prev_x:
            self._request_rest()
        self._prev_b, self._prev_y = snap.right.secondary, snap.left.secondary
        self._prev_x = snap.left.primary
        if first:
            logger.info(
                "VR 入力を受け始めた（left pose=%s right pose=%s head=%s）",
                snap.left.has_pose,
                snap.right.has_pose,
                snap.head.has_pose,
            )

    def _request_rest(self) -> None:
        """X で、追従を切って腕と head を rest へ戻す。戻る途中で A を押すと追従に切り替わる。"""
        if self._rest is None:
            logger.info("X を受けたが、--teleop.rest_path が無いので rest へ戻さない")
            return
        if not self._record.rest_allowed():
            logger.info("補正のあいだではないので X を無視した")
            return
        logger.info("X を受けたので rest へ戻す")
        self._reset_started.set()

    def _gripper_value(self, trigger: float) -> float:
        return trigger_to_gripper(
            trigger,
            open_value=self.config.gripper_open,
            closed_value=self.config.gripper_closed,
        )

    def _ik_limits_rad(self) -> dict[str, tuple[float, float]]:
        """キャリブレーションの可動域を、URDF の関節角（rad）の上下限へ換算する。"""
        out: dict[str, tuple[float, float]] = {}
        for side in SIDES:
            for joint in IK_JOINTS:
                motor = arm_motor(side, joint)
                if motor not in self._limits:
                    continue
                lim = self._limits[motor]
                out[ik_joint_name(side, joint)] = (math.radians(lim.lower_deg), math.radians(lim.upper_deg))
        return out

    def _q_from_motor_deg(self, motor_deg: dict[str, float]) -> np.ndarray:
        assert self._q is not None
        q = np.array(self._q, dtype=float)
        for side in SIDES:
            for joint in IK_JOINTS:
                motor = arm_motor(side, joint)
                if motor in motor_deg:
                    q[self._joint_index[ik_joint_name(side, joint)]] = math.radians(motor_deg[motor])
        return q

    def _push_frames(self) -> None:
        interval = 1.0 / self.config.fps
        warned = False
        while not self._frames_stop.wait(interval):
            try:
                self._push_head_frame()
            except Exception:
                # 映像が止まっても操作は続ける。原因を追えるよう初回だけ出す
                if not warned:
                    logger.warning("Quest へ映像を送れなかった", exc_info=True)
                    warned = True

    def _push_head_frame(self) -> None:
        """観測のカメラの画像が新しくなっていれば、1 枚に並べて Quest へ渡す。"""
        if self._head_video is None:
            return
        try:
            from lerobot_robot_xlerobot_book import observation_bus
        except ImportError:
            return
        frames = observation_bus.latest_frames()
        # 3 台の画像は同じ観測で入れ替わるので、head か最初の手首が同じなら新しい画像は無い
        newest = next((frames[n] for n in VIEW_CAMERAS if n in frames), None)
        if newest is None or newest is self._last_head_frame:
            return
        self._last_head_frame = newest
        view = compose_view(frames)
        status = self._record.status(time.monotonic())
        if status is not None:
            view = draw_status(view, *status)
        self._head_video.put_frame(bgr_for_teleop_xr(view))

    def _observed_pose_deg(self) -> dict[str, float] | None:
        try:
            from lerobot_robot_xlerobot_book import observation_bus
        except ImportError:
            return None
        obs = observation_bus.latest()
        if not obs or any(f"{m}.pos" not in obs for m in MOTOR_NAMES):
            return None
        return from_robot_units({m: obs[f"{m}.pos"] for m in MOTOR_NAMES}, self.calibration)

    def _resync_from_robot(self) -> None:
        """engage の基準を、ロボットの実測へ合わせる。実測が無ければ直近の指令のまま。"""
        observed = self._observed_pose_deg()
        if observed is None:
            return
        self._hold_deg = observed
        self._q = self._q_from_motor_deg(observed)


# Quest に出す 1 枚の大きさ。head を上の中央に大きく、左右の手首を下に並べる（どれも 4:3）
VIEW_SIZE = (1280, 720)
HEAD_BOX = (560, 420)
WRIST_BOX = (400, 300)
VIEW_CAMERAS = ("head", "left_wrist", "right_wrist")


def compose_view(frames: dict[str, np.ndarray]) -> np.ndarray:
    """head と左右の手首のカメラを 1 枚に並べる。手首のカメラが無ければ head をそのまま返す。"""
    import cv2

    if "left_wrist" not in frames and "right_wrist" not in frames:
        return frames["head"]
    width, height = VIEW_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    places = {
        "head": (HEAD_BOX, (width - HEAD_BOX[0]) // 2, 0),
        "left_wrist": (WRIST_BOX, width // 2 - WRIST_BOX[0], HEAD_BOX[1]),
        "right_wrist": (WRIST_BOX, width // 2, HEAD_BOX[1]),
    }
    for name, (box, x, y) in places.items():
        frame = frames.get(name)
        if frame is not None:
            canvas[y : y + box[1], x : x + box[0]] = cv2.resize(frame, box, interpolation=cv2.INTER_AREA)
    return canvas


def draw_status(view: np.ndarray, title: str, detail: str) -> np.ndarray:
    """記録の状態を左上に書く。REC は赤、それ以外は黄色。"""
    import cv2

    out = view.copy()
    color = (230, 40, 40) if title == "REC" else (240, 180, 20)
    cv2.circle(out, (40, 52), 18, color, -1)
    cv2.putText(out, title, (72, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4, cv2.LINE_AA)
    cv2.putText(out, detail, (24, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def idle_reason(snap: VRSnapshot) -> str:
    """追従が切れた理由。EngageStateMachine が IDLE へ移る条件と同じ順で見る。"""
    lost = [side for side, c in (("左", snap.left), ("右", snap.right)) if not c.has_pose]
    if lost:
        return f"{'と'.join(lost)}のコントローラのトラッキングを失った"
    if snap.right.primary:
        return "右の A が押された"
    return "理由が分からない"


def bgr_for_teleop_xr(frame: np.ndarray) -> np.ndarray:
    """観測の画像を、teleop_xr が期待する BGR にする。

    ホストはカメラの RGB をそのまま JPEG にする。OpenCV の符号化は BGR 前提なので、
    クライアントの imdecode が返す配列の並びは RGB になる。
    teleop_xr は受け取った配列を BGR とみなして RGB へ直してから Quest へ出す。
    """
    if getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        return frame
    return np.ascontiguousarray(frame[:, :, ::-1])


def _engage_ik_controller(robot: Any, solver: Any) -> Any:
    """teleop_xr の IKController から、engage の判定だけを外したもの。

    IKController は両 grip の押下をデッドマンにして自分で engage を決める。
    ここでは EngageStateMachine が決め、step は追従中にだけ呼ぶので、判定はいつも通す。
    それ以外（スナップショット、相対変位の合成、ソルバ呼び出し）は無改造で使う。
    """
    from teleop_xr.ik.controller import IKController

    class EngageIKController(IKController):
        def _check_deadman(self, state: Any) -> bool:
            return True

    return EngageIKController(robot, solver)
