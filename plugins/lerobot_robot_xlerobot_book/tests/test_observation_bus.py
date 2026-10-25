"""get_observation を包むと、最後の観測のスカラーだけが latest() から読める。"""

import numpy as np
from lerobot_robot_xlerobot_book import observation_bus


class FakeRobot:
    def get_observation(self):
        return {"left_arm_shoulder_pan.pos": 12.5, "x.vel": 0.0, "head": np.zeros((2, 2, 3))}


def test_install_publishes_scalars_and_keeps_image_references():
    observation_bus.install(FakeRobot)
    obs = FakeRobot().get_observation()
    assert "head" in obs  # 呼び出し側には元の観測がそのまま返る
    latest = observation_bus.latest()
    assert latest == {"left_arm_shoulder_pan.pos": 12.5, "x.vel": 0.0}
    assert observation_bus.latest_frames()["head"] is obs["head"]
