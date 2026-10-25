"""ロボットが最後に返した観測を、同じプロセスの Teleoperator から読めるようにする。

lerobot v0.6.1 の lerobot-teleoperate / lerobot-record のループは、
`teleop.send_feedback(obs)` を robot.name == "unitree_g1" のときしか呼ばない。
そのため robot 側の get_observation を包み、最後の観測を次の 3 つに分けて置く。

- latest(): 関節角などの数値。VR テレオペが engage の瞬間の基準に使う
- latest_frames(): ndarray の値（カメラの画像など）。コピーせず参照を置く
- latest_robot(): 観測を出したロボット。--teleop.type=xlerobot_vr_official が公式の get_action へ渡す
  弱参照にして、ロボットの寿命は延ばさない
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

import numpy as np

_lock = threading.Lock()
_latest: dict[str, float] | None = None
_latest_frames: dict[str, np.ndarray] = {}
_latest_robot: weakref.ref | None = None


def publish(obs: dict[str, Any], robot: object | None = None) -> None:
    global _latest, _latest_frames, _latest_robot
    scalars = {k: float(v) for k, v in obs.items() if isinstance(v, (int, float))}
    frames = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
    with _lock:
        _latest = scalars
        _latest_frames = frames
        if robot is not None:
            _latest_robot = weakref.ref(robot)


def latest() -> dict[str, float] | None:
    """最後の観測のスカラー部分（`left_arm_shoulder_pan.pos` など）。無ければ None"""
    with _lock:
        return None if _latest is None else dict(_latest)


def latest_frames() -> dict[str, np.ndarray]:
    """最後の観測の ndarray の値（カメラの画像など）。配列はコピーしない。無ければ空。"""
    with _lock:
        return dict(_latest_frames)


def latest_robot() -> Any | None:
    """最後に観測を出したロボット。無いか、すでに破棄されていれば None"""
    with _lock:
        return None if _latest_robot is None else _latest_robot()


def install(robot_cls: type) -> None:
    """robot_cls.get_observation を包んで、戻り値をここへ流す。"""
    original = robot_cls.get_observation

    def get_observation(self, *args, **kwargs):
        obs = original(self, *args, **kwargs)
        publish(obs, self)
        return obs

    get_observation.__doc__ = original.__doc__
    robot_cls.get_observation = get_observation
