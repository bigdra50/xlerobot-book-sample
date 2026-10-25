"""engage: A ボタンのトグルと grip の一時停止、トラッキングロスの扱い。"""

from __future__ import annotations

import pytest
from lerobot_teleoperator_xlerobot_quest.contracts import Controller, EngagePhase, HeadPose, VRSnapshot
from lerobot_teleoperator_xlerobot_quest.engage import EngageEvent, EngageStateMachine


def snap(
    *,
    a: bool = False,
    l_sq: bool = False,
    r_sq: bool = False,
    l_pose: bool = True,
    r_pose: bool = True,
) -> VRSnapshot:
    return VRSnapshot(
        left=Controller(squeeze=l_sq, has_pose=l_pose),
        right=Controller(squeeze=r_sq, primary=a, has_pose=r_pose),
        head=HeadPose(),
    )


def test_rising_edge_of_a_engages_with_rebase():
    sm = EngageStateMachine()
    ev = sm.update(snap(a=True))
    assert ev.phase is EngagePhase.ENGAGED
    assert ev.rebase is True
    assert ev.engaged is True


def test_release_then_press_toggles_back_to_idle():
    sm = EngageStateMachine()
    sm.update(snap(a=True))
    sm.update(snap(a=False))
    ev = sm.update(snap(a=True))
    assert ev.phase is EngagePhase.IDLE
    assert ev.rebase is False


def test_grip_pauses_and_release_rebases():
    sm = EngageStateMachine()
    sm.update(snap(a=True))
    ev = sm.update(snap(r_sq=True))
    assert ev == EngageEvent(phase=EngagePhase.PAUSED, rebase=False)
    ev = sm.update(snap(r_sq=True))
    assert ev.phase is EngagePhase.PAUSED
    ev = sm.update(snap())
    assert ev == EngageEvent(phase=EngagePhase.ENGAGED, rebase=True)
    ev = sm.update(snap())
    assert ev == EngageEvent(phase=EngagePhase.ENGAGED, rebase=False)


@pytest.mark.parametrize("lost", ["left", "right"])
def test_tracking_loss_while_engaged_goes_idle(lost):
    sm = EngageStateMachine()
    sm.update(snap(a=True))
    kwargs = {"l_pose": lost != "left", "r_pose": lost != "right"}
    ev = sm.update(snap(**kwargs))
    assert ev == EngageEvent(phase=EngagePhase.IDLE, rebase=False)
    # 復帰しても勝手に再開しない。A を押し直すまで IDLE
    ev = sm.update(snap())
    assert ev.phase is EngagePhase.IDLE
