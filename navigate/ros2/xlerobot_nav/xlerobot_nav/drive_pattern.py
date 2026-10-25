"""評価用のノード。融合した /odom を見ながら /cmd_vel を出し、台車を決まった形で走らせる。

pattern=outback: その場で 180°、前へ distance、その場で 180°、前へ distance。
pattern=strafe: 向きを変えずに、左へ distance、右へ 2 distance、左へ distance を 3 回。
旋回と直進は、止めずに続けて指令する（Nav2 の出す指令に近い形）。残りが小さくなったら速さを落とす。
終わったら、始めと終わりの /odom、/wheel/odom、ARKit の姿勢を JSON で out へ書く。

安全の約束は odom_driver にある。並進の向きは補正しない（評価のため）。
"""

from __future__ import annotations

import json

import rclpy

from xlerobot_nav.odom_driver import OdomDriver


class DrivePattern(OdomDriver):
    def __init__(self) -> None:
        super().__init__("drive_pattern")
        self.pattern = str(self.declare_parameter("pattern", "outback").value)
        self.distance = float(self.declare_parameter("distance", 0.6).value)
        self.speed = float(self.declare_parameter("speed", 0.1).value)
        self.turn_speed = float(self.declare_parameter("turn_speed", 20.0).value)
        self.out = str(self.declare_parameter("out", "/out/drive_pattern.json").value)

    def run(self) -> dict:
        self.wait_inputs()
        self.hold(2.0)
        result = {
            "pattern": self.pattern,
            "distance": self.distance,
            "speed": self.speed,
            "turn_speed": self.turn_speed,
            "start": {k: t.snapshot() for k, t in self.tracks.items()},
        }
        d = self.distance
        steps = {
            "outback": [("turn", 180.0), ("forward", d), ("turn", 180.0), ("forward", d)],
            "strafe": [("strafe", d), ("strafe", -2 * d), ("strafe", d)] * 3,
        }[self.pattern]
        try:
            for kind, amount in steps:
                self.get_logger().info(f"{kind} {amount}")
                if kind == "turn":
                    self.turn(amount)
                else:
                    self.translate(amount, sideways=kind == "strafe")
        finally:
            self.hold(2.0)
            result["end"] = {k: t.snapshot() for k, t in self.tracks.items()}
            with open(self.out, "w") as f:
                json.dump(result, f, indent=1)
            self.get_logger().info(f"wrote {self.out}")
        return result


def main() -> None:
    rclpy.init()
    node = DrivePattern()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.hold(0.5)
        node.destroy_node()
        rclpy.try_shutdown()
