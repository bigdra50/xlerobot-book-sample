"""safety: 速度制限、入力の鮮度、関節上下限。"""

from __future__ import annotations

from lerobot_teleoperator_xlerobot_quest.contracts import JointLimits
from lerobot_teleoperator_xlerobot_quest.safety import (
    SlewLimiter,
    StaleGate,
    clamp_to_limits,
)


class TestSlewLimiter:
    def test_large_step_is_clipped_both_directions(self):
        lim = SlewLimiter({"a": 5.0})
        assert lim.step({"a": 10.0}, {"a": 100.0}) == {"a": 15.0}
        assert lim.step({"a": 10.0}, {"a": -100.0}) == {"a": 5.0}


class TestStaleGate:
    def test_no_input_is_stale(self):
        assert StaleGate(0.3).is_fresh(0.0) is False
        assert StaleGate(0.3).is_fresh(100.0) is False

    def test_fresh_within_timeout(self):
        g = StaleGate(0.3)
        g.note_input(10.0)
        assert g.is_fresh(10.0) is True
        assert g.is_fresh(10.29) is True


def test_clamp_to_limits():
    limits = {"a": JointLimits(-90.0, 90.0), "b": JointLimits(0.0, 10.0)}
    out = clamp_to_limits({"a": 120.0, "b": -3.0, "c": 999.0}, limits)
    assert out == {"a": 90.0, "b": 0.0, "c": 999.0}
