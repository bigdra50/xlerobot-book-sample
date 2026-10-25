# 元: Vector-Wangel/XLeRobot software/plugins/lerobot_robot_xlerobot/lerobot_robot_xlerobot/xlerobot.py の XLerobot._body_to_wheel_raw（Apache-2.0）
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# 変更点: 同じ式と定数を numpy なしで書いた
"""オムニベースを単発コマンドで駆動し、走行方向だけを確かめる。

公式のロボットクラス XLerobot は connect で 17 台すべてを要求するので、
ホイールだけを先に回すことができない。ここは bus2（右腕 ID1-6 + ホイール ID7-9）を
直接開いて、ホイール 3 台にだけ Goal_Velocity を書く。腕のトルクには触れない。

車体速度から 3 輪の速度への変換は公式 XLerobot._body_to_wheel_raw と同じ式・同じ定数
（取付角 150 / -90 / 30 度、車輪半径 0.05m、車体半径 0.125m）。方向の定義も公式に合わせる。

  forward      = +x（後輪 ID8 を置いた側の反対向き）
  strafe-left  = +y（前方に向かって左）
  turn-left    = +theta（上から見て反時計回り）

後輪 ID8 の駆動方向は横向きなので、純粋な前進では ID8 は回らない（raw=0 が正常）。

初回は必ず輪を浮かせて回転方向だけを見る。床に置くのは方向が合ってから。
USB ケーブルと 12V 配線を巻き込む事故は、浮かせている限り起きない。

Usage:
    python drive_base.py --port /dev/ttyACM1 --move forward
    python drive_base.py --port /dev/ttyACM1 --move turn-left --duration 2

Exit code: 駆動できたら 0、ID7-9 が揃わない・ポートが開けないなら 1。
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Final

WHEEL_IDS: Final[dict[str, int]] = {"base_left_wheel": 7, "base_back_wheel": 8, "base_right_wheel": 9}

# 公式 XLerobot._body_to_wheel_raw の定数。変えると走行距離と旋回量がずれる
WHEEL_RADIUS_M: Final[float] = 0.05
BASE_RADIUS_M: Final[float] = 0.125
MOUNT_ANGLES_DEG: Final[tuple[float, ...]] = (150.0, -90.0, 30.0)  # np.array([240, 0, 120]) - 90
STEPS_PER_DEG: Final[float] = 4096.0 / 360.0
MAX_WHEEL_RAW: Final[int] = 3000  # 公式 _body_to_wheel_raw の max_raw と同じ

# 安全上限。浮かせた輪の方向確認に速さは要らない。この値で飽和させる
MAX_LINEAR_MPS: Final[float] = 0.2
MAX_ANGULAR_DPS: Final[float] = 60.0
MAX_DURATION_S: Final[float] = 3.0

# (x[m/s 係数], y[m/s 係数], theta[deg/s 係数])
MOVES: Final[dict[str, tuple[float, float, float]]] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "strafe-left": (0.0, 1.0, 0.0),
    "strafe-right": (0.0, -1.0, 0.0),
    "turn-left": (0.0, 0.0, 1.0),
    "turn-right": (0.0, 0.0, -1.0),
}


def body_to_wheel_raw(x: float, y: float, theta_dps: float) -> dict[str, int]:
    """車体速度 (m/s, m/s, deg/s) を 3 輪の Goal_Velocity raw へ変換する。

    出典: XLeRobot 公式 software/plugins/lerobot_robot_xlerobot/lerobot_robot_xlerobot/xlerobot.py の _body_to_wheel_raw。
    numpy を使わずに書き下しただけで、式と定数は同一。
    """
    theta_rad = math.radians(theta_dps)
    wheel_dps: dict[str, float] = {}
    for name, angle_deg in zip(WHEEL_IDS, MOUNT_ANGLES_DEG, strict=True):
        angle = math.radians(angle_deg)
        linear_mps = math.cos(angle) * x + math.sin(angle) * y + BASE_RADIUS_M * theta_rad
        wheel_dps[name] = math.degrees(linear_mps / WHEEL_RADIUS_M)
    # 1 輪でも上限を超えたら 3 輪を同じ比率で落とす。1 輪だけ切ると走行方向が変わる
    peak = max(abs(v) * STEPS_PER_DEG for v in wheel_dps.values())
    scale = MAX_WHEEL_RAW / peak if peak > MAX_WHEEL_RAW else 1.0
    return {name: round(v * STEPS_PER_DEG * scale) for name, v in wheel_dps.items()}


def main() -> int:
    # lerobot は main でだけ import する。navigate/gate のテストが lerobot の無い環境でこのモジュールを import する
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="bus2（右腕 + ホイール）のポート")
    parser.add_argument("--move", required=True, choices=sorted(MOVES))
    parser.add_argument("--speed", type=float, default=0.1, help="直進速度 m/s（既定 0.1）")
    parser.add_argument("--turn-speed", type=float, default=30.0, help="旋回速度 deg/s（既定 30）")
    parser.add_argument("--duration", type=float, default=1.0, help="駆動時間 秒（既定 1）")
    args = parser.parse_args()

    speed = min(abs(args.speed), MAX_LINEAR_MPS)
    turn = min(abs(args.turn_speed), MAX_ANGULAR_DPS)
    duration = min(abs(args.duration), MAX_DURATION_S)
    kx, ky, kt = MOVES[args.move]
    targets = body_to_wheel_raw(kx * speed, ky * speed, kt * turn)

    motors = {name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for name, i in WHEEL_IDS.items()}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect(handshake=False)
    found = frozenset(bus.broadcast_ping() or {})
    if not set(WHEEL_IDS.values()) <= found:
        # bus1（左腕 ID1-6、head ID7-8）を指定すると、ID7-8 は head になる。
        # 速度 0 も閉じるときの脱力も head への書き込みなので、何も書かずに閉じる
        bus.disconnect(disable_torque=False)
        print(f"{args.port}: ID 7,8,9 が揃わない（応答: {sorted(found)}）。bus2 のポートか確認する")
        return 1
    try:
        # Operating_Mode は EEPROM 側。Lock を落とさないと書き込みが効かない。
        # disable_torque が Torque_Enable=0 と Lock=0 の両方を書く
        bus.disable_torque()
        for name in motors:
            bus.write("Operating_Mode", name, 1, normalize=False)  # 1 = VELOCITY
        bus.enable_torque()
        print(f"{args.move}: {duration:.1f} 秒")
        for name, value in targets.items():
            print(f"  {name:18s} {value:6d}")
        bus.sync_write("Goal_Velocity", targets)
        time.sleep(duration)
    finally:
        # 例外でも Ctrl+C でも必ず止めてから脱力する。回りっぱなしで抜けさせない
        try:
            bus.sync_write("Goal_Velocity", dict.fromkeys(motors, 0), num_retry=5)
        finally:
            bus.disconnect(disable_torque=True)
    print("停止・脱力した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
