"""公式の XLerobotVRTeleop を、lerobot-teleoperate から --teleop.type=xlerobot_vr_official で使う入口。

公式の Teleoperator は get_action(robot_obs, robot) と、ロボットの観測とロボットを引数に取る。
lerobot v0.6.1 の標準のループは get_action() を引数なしで呼ぶため、そのままでは空の辞書しか返らない。
公式は LeRobot の record.py を手直しした版から呼んでいる。
ここでは lerobot_robot_xlerobot_book.observation_bus から観測とロボットを読み、公式の get_action へ渡す。

起動の前に、XLeVR の場所を --teleop.xlevr_path の値で上書きする。
公式の vr_monitor.py は作者の絶対パスを持ち、公式の設定の xlevr_path を読まないため。

公式の動かし方（XLeVR の入力、手の移動の倍率、解析 IK、ベースの割り当て）は変えない。
変えるのは次の3つで、どれも腕の動かし方には触れない。

- 起動直後に腕がゼロ姿勢へ移る間の変化を、start_max_step で丸める
- 右コントローラの入力が途絶えたら、ベースの速度を0にする
- 公式が空の辞書を返した周は、直前の指令を保つ（ベースだけ0）
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_xlerobot_vr_official import XLeRobotVROfficialConfig
from .contracts import ACTION_KEYS, BASE_KEYS
from .safety import SlewLimiter

logger = logging.getLogger(__name__)

POSITION_KEYS: tuple[str, ...] = tuple(k for k in ACTION_KEYS if k not in BASE_KEYS)
BASE_STOP: dict[str, float] = dict.fromkeys(BASE_KEYS, 0.0)


class XLeRobotVROfficial(Teleoperator):
    config_class = XLeRobotVROfficialConfig
    name = "xlerobot_vr_official"

    def __init__(self, config: XLeRobotVROfficialConfig):
        super().__init__(config)
        self.config = config
        self._official: Any = None
        self._ramp = (
            SlewLimiter(dict.fromkeys(POSITION_KEYS, config.start_max_step))
            if config.start_max_step > 0
            else None
        )
        self._last_action: dict[str, float] | None = None
        # id() ではなくオブジェクトを持って比べる。持っている間は、新しい入力が同じアドレスに作られない
        self._right_goal: Any = None
        self._right_goal_t = 0.0

    @property
    def action_features(self) -> dict:
        # 公式の action_features はベースを "base_action" の1項目にまとめていて、ロボットの17キーと合わない
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._official is not None and self._official.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._official is not None and self._official.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} はすでに接続している")
        try:
            from lerobot_robot_xlerobot_book import observation_bus  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "xlerobot_vr_official には lerobot_robot_xlerobot_book（--robot.type=xlerobot_book_client）が要る。"
                "ロボットの観測を受け取る経路がそこにしか無い"
            ) from e

        # 公式の VR モニタは、依存が欠けていても初期化を10秒繰り返してから、原因を書かずに失敗する
        missing = [m for m in ("scipy", "websockets", "yaml") if importlib.util.find_spec(m) is None]
        if missing:
            raise RuntimeError(
                f"XLeVR の依存が無い: {', '.join(missing)}。"
                "uv pip install -e 'plugins/lerobot_teleoperator_xlerobot_quest[vr,xlevr]' で入れる"
            )

        xlevr = Path(self.config.xlevr_path).expanduser()
        if not (xlevr / "xlevr").is_dir():
            raise FileNotFoundError(
                f"XLeVR が見つからない: {xlevr}。"
                "install.sh で公式リポジトリを取得し、--teleop.xlevr_path で指す"
            )
        # 公式の VR モニタは XLeVR の場所を作者の絶対パスで持ち、起動時にそこへ os.chdir する。
        # 証明書（cert.pem、key.pem）と config.yaml を作業ディレクトリから読むためで、chdir は残す。
        # 記録の出力先を相対パスで指すと chdir の影響を受けるので、試すのは lerobot-teleoperate にとどめる
        from lerobot_teleoperator_xlerobot_vr import vr_monitor

        vr_monitor.XLEVR_PATH = str(xlevr)
        from lerobot_teleoperator_xlerobot_vr.xlerobot_vr import XLerobotVRTeleop

        self._official = XLerobotVRTeleop(self.config)
        # 公式の calibrate は、モーターのキャリブレーションではなく、最初の観測から腕の制御の状態を作る処理。
        # 観測は get_action の周でしか手に入らないので、ここでは渡さずに最初の get_action で呼ぶ
        self._official.connect(calibrate=False)

    def calibrate(self) -> None:
        """何もしない。モーターのキャリブレーションは robot 側が持ち、公式の calibrate は get_action で呼ぶ。"""

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} は接続していない")
        from lerobot_robot_xlerobot_book import observation_bus

        obs = observation_bus.latest()
        robot = observation_bus.latest_robot()
        if obs is None or robot is None:
            # 標準のループは get_action の前に get_observation を呼ぶので、ここへ来るのはループの外から呼んだときだけ
            raise RuntimeError("ロボットの観測がまだ無い。--robot.type=xlerobot_book_client と一緒に使う")

        if not self._official.is_calibrated:
            self._official.calibrate(obs)

        action = self._official.get_action(obs, robot)
        if not action:
            return self._hold(obs)
        if self._right_input_is_stale():
            action.update(BASE_STOP)
        if self._ramp is not None:
            action = self._ramp_from_last(action, obs)
        self._last_action = action
        return action

    def _hold(self, obs: dict[str, float]) -> dict[str, float]:
        """公式が空の辞書を返した周（例外を握り潰したとき）は、直前の指令を保ち、ベースだけ止める。"""
        held = (
            dict(self._last_action) if self._last_action else {k: obs[k] for k in POSITION_KEYS if k in obs}
        )
        held.update(BASE_STOP)
        self._last_action = held
        return held

    def _right_input_is_stale(self) -> bool:
        """右コントローラの入力が stale_timeout_s 以上更新されていなければ True。

        XLeVR は Quest からのメッセージごとに新しい ControlGoal を作るので、
        同じオブジェクトが残り続けていれば入力が途絶えている。
        一度も入力が無いときは、ベースの入力（スティック）も無いので止めても変わらない。
        """
        if self.config.stale_timeout_s <= 0:
            return False
        goal = self._official.vr_monitor.get_latest_goal_nowait("right")
        now = time.monotonic()
        if goal is None:
            return True
        if goal is not self._right_goal:
            self._right_goal = goal
            self._right_goal_t = now
            return False
        return now - self._right_goal_t > self.config.stale_timeout_s

    def _ramp_from_last(self, action: dict[str, float], obs: dict[str, float]) -> dict[str, float]:
        """腕がゼロ姿勢へ移り終えるまで、腕と head の変化を start_max_step で丸める。

        丸めた指令が公式の指令に追いついたら、以後は丸めない。
        腕を動かしている最中まで丸めると、公式の動かし方の感触が変わってしまう。
        """
        prev = self._last_action if self._last_action is not None else obs
        step = self.config.start_max_step
        if all(abs(action[k] - prev[k]) <= step for k in POSITION_KEYS if k in action and k in prev):
            self._ramp = None
            logger.info("ゼロ姿勢へ移り終えた。ここからは公式の指令をそのまま送る")
            return action
        return self._ramp.step(prev, action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} は接続していない")
        # 公式の disconnect は VR モニタのスレッドを止めない（daemon スレッドで、プロセスの終了とともに止まる）
        self._official.disconnect()
        self._official = None
