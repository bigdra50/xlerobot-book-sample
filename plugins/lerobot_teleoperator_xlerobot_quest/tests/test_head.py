"""head: HMD の向きの差分を head の pan / tilt に足す。符号は実機で確認する前提。"""

from __future__ import annotations

import pytest
from lerobot_teleoperator_xlerobot_quest.contracts import HeadPose
from lerobot_teleoperator_xlerobot_quest.head import HeadFollower


def head(yaw: float = 0.0, pitch: float = 0.0, has_pose: bool = True) -> HeadPose:
    return HeadPose(yaw_deg=yaw, pitch_deg=pitch, has_pose=has_pose)


def test_yaw_delta_adds_to_pan():
    hf = HeadFollower()
    hf.rebase(head(yaw=20.0), pan_deg=5.0, tilt_deg=0.0)
    pan, tilt = hf.update(head(yaw=30.0))
    assert pan == pytest.approx(15.0)
    assert tilt == pytest.approx(0.0)


def test_yaw_wraps_across_pm180():
    hf = HeadFollower()
    hf.rebase(head(yaw=170.0), pan_deg=0.0, tilt_deg=0.0)
    pan, _ = hf.update(head(yaw=-170.0))  # 右へ 20 度回ったのであって 340 度ではない
    assert pan == pytest.approx(20.0)
    hf.rebase(head(yaw=-170.0), pan_deg=0.0, tilt_deg=0.0)
    pan, _ = hf.update(head(yaw=170.0))
    assert pan == pytest.approx(-20.0)
