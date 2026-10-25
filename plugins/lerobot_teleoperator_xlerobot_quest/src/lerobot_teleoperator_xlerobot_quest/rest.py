"""エピソードを終えたあと、腕と head を rest の姿勢へ戻す。

全関節を同時に、決まった時間をかけて動かす。ゆっくり動き出し、ゆっくり止まる（smoothstep）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .contracts import MOTOR_NAMES, JointLimits

logger = logging.getLogger(__name__)

# 距離によらず、この時間で目標へ着く指令を出す
DURATION_S = 2.5
# 指令が目標へ着いたあとも、この時間は止まった関節を見張る。補間の終わりぎわに阻まれた関節を拾う
SETTLE_S = 1.0
# 戻り終えたときに、これより離れている関節をログに出す
REACHED_DEG = 2.0
# 指令から 15 度以上離れたまま、0.5 秒で 2 度も動かない関節は、何かに阻まれている。
# サーボの過電流保護（2 秒）が働く前に、その関節だけ今の位置で保持へ切り替える。
# 低いゲインと重力による追従の遅れは数度なので、15 度なら取り違えない
STALL_ERR_DEG = 15.0
STALL_PROGRESS_DEG = 2.0
STALL_HOLD_S = 0.5
# 可動域の端は機構に当たる位置なので、目標は端から内側に置く（host の 50 raw）
RANGE_MARGIN_DEG = 4.4


def load_rest(path: Path) -> dict[str, float]:
    """host の --rest.path と同じ JSON（モーター名 → キャリブレーションの範囲の中央を 0 とした度）。"""
    return {m: float(v) for m, v in json.loads(Path(path).read_text()).items() if m in MOTOR_NAMES}


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class RestReturn:
    def __init__(self, rest_deg: dict[str, float], limits: dict[str, JointLimits], gripper_closed: float):
        self._goal: dict[str, float] = {}
        for m, deg in rest_deg.items():
            if m.endswith("gripper"):
                self._goal[m] = gripper_closed
            elif m in limits:
                lim = limits[m]
                self._goal[m] = min(
                    lim.upper_deg - RANGE_MARGIN_DEG, max(lim.lower_deg + RANGE_MARGIN_DEG, deg)
                )
        self._active = False
        self._start: dict[str, float] = {}
        self._start_t: float | None = None
        self._held: dict[str, float] = {}
        self._progress: dict[str, tuple[float, float]] = {}

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._active = True
        self._start_t = None
        self._held.clear()
        self._progress.clear()

    def cancel(self) -> None:
        self._active = False

    def step(
        self, command: dict[str, float], observed: dict[str, float] | None, now: float
    ) -> dict[str, float]:
        """直近の指令から rest の姿勢へ補間した指令を返す。"""
        if not self._active:
            return command
        if self._start_t is None:
            self._start_t = now
            self._start = {m: command[m] for m in self._goal}
        elapsed = now - self._start_t
        blend = smoothstep(min(elapsed / DURATION_S, 1.0))
        out = dict(command)
        for m, goal in self._goal.items():
            out[m] = self._start[m] + blend * (goal - self._start[m])
            if observed is not None and m not in self._held:
                self._watch_stall(m, out[m], observed[m], now)
            out[m] = self._held.get(m, out[m])
        if elapsed >= DURATION_S + SETTLE_S:
            self._active = False
            if observed is not None:
                missed = {
                    m: round(observed[m] - goal, 1)
                    for m, goal in self._goal.items()
                    if abs(observed[m] - goal) > REACHED_DEG
                }
                if missed:
                    logger.info("rest から離れて止まった関節（目標との差）: %s", missed)
        return out

    def _watch_stall(self, motor: str, command: float, position: float, now: float) -> None:
        if abs(command - position) <= STALL_ERR_DEG:
            self._progress.pop(motor, None)
            return
        last = self._progress.get(motor)
        if last is None or abs(position - last[1]) >= STALL_PROGRESS_DEG:
            self._progress[motor] = (now, position)
        elif now - last[0] >= STALL_HOLD_S:
            self._held[motor] = position
            logger.warning("%s が %.1f 度で止まったので、押すのをやめて保持する", motor, position)
