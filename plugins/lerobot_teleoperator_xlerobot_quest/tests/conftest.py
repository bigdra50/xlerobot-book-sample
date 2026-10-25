"""テスト共通の材料。実機・ヘッドセット・ネットワークは要らない。"""

from __future__ import annotations

import pytest


@pytest.fixture
def calibration_dict() -> dict[str, dict[str, int]]:
    """公式 17 台 1 ファイル形式と同じキー。値は架空だが形は実物と同じ。

    range は全関節で 1024..3072（半回転中央 2048 を中心に ±90 度）、
    gripper だけ 2048..3072、wrist_roll は全周。
    """
    from lerobot_teleoperator_xlerobot_quest.contracts import MOTOR_NAMES

    cal = {}
    for i, name in enumerate(MOTOR_NAMES, start=1):
        if name.endswith("gripper"):
            rng = (2048, 3072)
        elif name.endswith("wrist_roll"):
            rng = (0, 4095)
        else:
            rng = (1024, 3072)
        cal[name] = {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": rng[0], "range_max": rng[1]}
    for i, name in enumerate(("base_left_wheel", "base_back_wheel", "base_right_wheel"), start=15):
        cal[name] = {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
    return cal
