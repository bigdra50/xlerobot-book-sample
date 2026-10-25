"""gripper: トリガーの押し込み量から gripper の指令へ。握るほど開く。"""

from __future__ import annotations

from lerobot_teleoperator_xlerobot_quest.gripper import trigger_to_gripper

OPEN, CLOSED = 100.0, 0.0


def grip(trigger: float) -> float:
    return trigger_to_gripper(trigger, open_value=OPEN, closed_value=CLOSED)


def test_released_trigger_closes():
    assert grip(0.0) == CLOSED


def test_full_trigger_opens():
    assert grip(1.0) == OPEN
