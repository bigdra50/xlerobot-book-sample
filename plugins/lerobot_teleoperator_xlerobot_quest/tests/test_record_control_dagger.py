"""lerobot-rollout の dagger を Quest のボタンで切り替える。

B は交代（ポリシーを止めて補正を始める、補正を終える）、Y は一時停止と再開。
dagger は phase を 1 回に 1 つしか進めないので、止まったログを見てから補正を始める。
"""

from __future__ import annotations

import logging

import pytest

dagger = pytest.importorskip("lerobot.rollout.strategies.dagger")

from lerobot_teleoperator_xlerobot_quest import record_control
from lerobot_teleoperator_xlerobot_quest.record_control import RecordControl

HELP = "pause_resume='space', correction='tab', upload='enter', ESC=stop"


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setattr(record_control, "DAGGER_PRESS_GAP_S", 0.0)
    started: list[bool] = []
    rc = RecordControl(on_reset=lambda: None, on_correction_start=lambda: started.append(True))
    rc.install()
    sent: list[str] = []
    dagger.create_key_listener(sent.append, controls_help=HELP)
    log = logging.getLogger(dagger.__name__)
    level = log.level
    log.setLevel(logging.INFO)
    yield rc, sent, started, log
    log.setLevel(level)
    rc.uninstall()


def _transition(log, old: str, new: str) -> None:
    log.info("Phase transition: %s -> %s", old, new)


def test_b_takes_over_then_finishes(control):
    rc, sent, started, log = control
    rc.press("right")
    assert sent == ["space"]
    _transition(log, "autonomous", "paused")
    assert sent == ["space", "tab"]
    _transition(log, "paused", "correcting")
    assert started == [True]
    rc.press("right")
    assert sent == ["space", "tab", "tab"]


def test_y_pauses_and_resumes_but_not_while_correcting(control):
    rc, sent, _, log = control
    rc.press("left")
    _transition(log, "autonomous", "paused")
    assert sent == ["space"]
    rc.press("left")
    assert sent == ["space", "space"]
    _transition(log, "paused", "autonomous")
    rc.press("right")
    _transition(log, "autonomous", "paused")
    _transition(log, "paused", "correcting")
    rc.press("left")
    assert sent == ["space", "space", "space", "tab"]


def test_uninstall_restores_dagger_listener():
    original = dagger.create_key_listener
    rc = RecordControl(on_reset=lambda: None)
    rc.install()
    assert dagger.create_key_listener is not original
    rc.uninstall()
    assert dagger.create_key_listener is original


def test_starts_paused_until_y(control):
    # 起動した時点では Quest をまだ被っていない。ポリシーは Y を押すまで動かさない
    _, sent, _, log = control
    log.info("DAgger corrections-only recording started (target: %d episodes)", 20)
    assert sent == ["space"]


def test_rest_only_while_correcting(control):
    # dagger が止まっているあいだやポリシーが動かしているあいだは get_action が呼ばれない。
    # そこで受けた X が、次の補正の始まりで腕を動かさないこと
    rc, _, _, log = control
    assert not rc.rest_allowed()
    rc.press("right")
    _transition(log, "autonomous", "paused")
    assert not rc.rest_allowed()
    _transition(log, "paused", "correcting")
    assert rc.rest_allowed()


def test_rest_allowed_outside_dagger():
    rc = RecordControl(on_reset=lambda: None)
    assert rc.rest_allowed()
