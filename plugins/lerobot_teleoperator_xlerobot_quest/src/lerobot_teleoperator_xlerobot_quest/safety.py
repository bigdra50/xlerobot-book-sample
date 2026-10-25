"""指令をモーターへ渡す直前の安全弁。速度制限、入力の鮮度、関節上下限。

IK の解が飛んだとき（特異点付近や到達不能な目標）に腕が一気に動くのを防ぐのが目的で、
軌道の滑らかさのためではない。
"""

from __future__ import annotations

from .contracts import JointLimits


class SlewLimiter:
    """1 ティックあたりの変化量をキーごとに上限で丸める。"""

    def __init__(self, max_step_by_key: dict[str, float]) -> None:
        for key, step in max_step_by_key.items():
            if step < 0:
                raise ValueError(f"{key}: max_step が負（{step}）")
        self._max_step = dict(max_step_by_key)

    def step(self, prev: dict[str, float], target: dict[str, float]) -> dict[str, float]:
        """max_step に無いキーは素通し。prev に無いキーは前回値が無いので target をそのまま返す。"""
        out: dict[str, float] = {}
        for key, want in target.items():
            step = self._max_step.get(key)
            if step is None or key not in prev:
                out[key] = want
                continue
            delta = want - prev[key]
            out[key] = prev[key] + max(-step, min(step, delta))
        return out


def max_step_from_speed(max_dps: float, fps: int) -> float:
    """度/秒を度/ティックへ。"""
    if fps <= 0:
        raise ValueError(f"fps は正の整数（{fps}）")
    return max_dps / fps


class StaleGate:
    """VR 入力が途絶えたことを検出する。一度も入力が無ければ stale。

    境界は含む（経過時間 == timeout はまだ fresh）。
    """

    def __init__(self, timeout_s: float) -> None:
        if timeout_s < 0:
            raise ValueError(f"timeout_s が負（{timeout_s}）")
        self._timeout_s = timeout_s
        self._last_t_s: float | None = None

    def note_input(self, t_s: float) -> None:
        self._last_t_s = t_s

    def age(self, t_now_s: float) -> float:
        """最後の入力からの経過時間。一度も入力が無ければ inf。"""
        return float("inf") if self._last_t_s is None else t_now_s - self._last_t_s

    def is_fresh(self, t_now_s: float) -> bool:
        if self._last_t_s is None:
            return False
        # t_now_s を取ったあとに別スレッドが note_input すると、経過時間は負になる。直前に入力があったので fresh
        return t_now_s - self._last_t_s <= self._timeout_s


def clamp_to_limits(values: dict[str, float], limits: dict[str, JointLimits]) -> dict[str, float]:
    """limits に無いキーは素通し。"""
    return {key: (limits[key].clamp(v) if key in limits else v) for key, v in values.items()}
