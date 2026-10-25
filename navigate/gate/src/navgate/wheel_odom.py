"""エンコーダの絶対位置の読みを順に受け取り、車輪だけのオドメトリを積み上げる。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from navgate.fusion import FusionParams, Pose2D, fuse_step
from navgate.kinematics import WHEEL_ORDER, unwrap_steps, wheel_steps_to_body


@dataclass
class WheelOdometry:
    yaw_scale: float = 1.0
    pose: Pose2D = field(default_factory=Pose2D)
    _last: dict[str, int] | None = None

    def update(self, pos: Mapping[str, int]) -> tuple[float, float, float] | None:
        """前の読みからの車体座標の変位を返し、姿勢を進める。最初の読みは基準にするだけで None を返す。

        読みのあいだに車輪が半周を超えて回ると向きを取り違えるので、読みは 50 Hz 程度で渡す。
        """
        now = {k: int(pos[k]) for k in WHEEL_ORDER}
        if self._last is None:
            self._last = now
            return None
        steps = {k: unwrap_steps(self._last[k], now[k]) for k in WHEEL_ORDER}
        self._last = now
        delta = wheel_steps_to_body(steps, self.yaw_scale)
        self.pose = fuse_step(self.pose, delta[0], delta[1], delta[2], None, FusionParams()).pose
        return delta
