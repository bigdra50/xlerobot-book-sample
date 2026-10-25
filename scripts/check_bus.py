"""XLeRobot の 2 バス 17 サーボを、読み取りだけで診断する。

broadcast ping でバス上に実在する ID を列挙し、期待する構成
（bus1: 左腕 ID1-6 + head ID7-8 / bus2: 右腕 ID1-6 + ホイール ID7-9）
と突き合わせて、電圧、温度、エラーフラグを報告する。

どのポートがどちらのバスかは、接続順ではなく応答した ID の構成から決める
（ID9 が居れば bus2、ID7 か ID8 だけなら bus1）。/dev/ttyACM の番号は挿抜と再起動で入れ替わる。
ID はバス内で一意であればよく、バスをまたいだ重複は問題にならない。

モーターへは何も書かない。トルクの状態も変えないので、姿勢を保持している腕は落ちない。

Usage:
    python check_bus.py                       # USB シリアルを自動列挙
    python check_bus.py --port /dev/ttyACM0 --port /dev/ttyACM1

Exit code: 17 台すべて健全なら 0、それ以外は 1。
"""

from __future__ import annotations

import argparse
from typing import Final

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from serial.tools import list_ports

MODEL: Final[str] = "sts3215"
# 1 本のバスに載る ID の上限（腕 ID1-6 + head ID7-8、または腕 ID1-6 + ホイール ID7-9）
MAX_BUS_ID: Final[int] = 9

# STS3215 の raw 値。電圧は 0.1V 単位（12V 系の実用範囲 11.5-12.6V）
VOLTAGE_RAW_MIN: Final[int] = 115
VOLTAGE_RAW_MAX: Final[int] = 126
TEMP_WARN_C: Final[int] = 50

REGISTERS: Final[tuple[str, ...]] = (
    "Present_Position",
    "Present_Voltage",
    "Present_Temperature",
    "Status",
    "Torque_Enable",
    "Operating_Mode",
)

# ID の物理的な意味。バスによって 7 以降の指す先が変わる。
# 名前は XLeRobot 公式の XLerobot クラス（software/plugins/lerobot_robot_xlerobot）に合わせた。
# 公式のログやキャリブレーション JSON に出てくるキーと突き合わせられる。
JOINT_NAMES: Final[dict[str, dict[int, str]]] = {
    "bus1": {
        1: "left_arm_shoulder_pan",
        2: "left_arm_shoulder_lift",
        3: "left_arm_elbow_flex",
        4: "left_arm_wrist_flex",
        5: "left_arm_wrist_roll",
        6: "left_arm_gripper",
        7: "head_motor_1 (pan)",
        8: "head_motor_2 (tilt)",
    },
    "bus2": {
        1: "right_arm_shoulder_pan",
        2: "right_arm_shoulder_lift",
        3: "right_arm_elbow_flex",
        4: "right_arm_wrist_flex",
        5: "right_arm_wrist_roll",
        6: "right_arm_gripper",
        # 公式名は base_left/back/right_wheel だが、確認作業ではチェーン順で呼ぶ。
        # カートへ固定する前は前後左右が物理的に定まらず、車体基準で呼ぶと作業者間で認識がずれる
        7: "base_left_wheel (chain 1st)",
        8: "base_back_wheel (chain 2nd)",
        9: "base_right_wheel (chain 3rd)",
    },
}


def classify_bus(found_ids: frozenset[int]) -> tuple[str, dict[int, str]]:
    """応答した ID の構成からバス種別を決める。ID1-6 だけなら判別できない。"""
    if 9 in found_ids:
        return "bus2", JOINT_NAMES["bus2"]
    if found_ids & {7, 8}:
        return "bus1", JOINT_NAMES["bus1"]
    return "unknown", {}


def usb_serial_ports() -> list[str]:
    """VID を持つ USB シリアルだけを返す（Bluetooth の仮想ポートを除く）。"""
    return sorted(p.device for p in list_ports.comports() if p.vid is not None)


def open_all_ids(port: str) -> FeetechMotorsBus:
    """ID1-9 を仮の名前 id1-id9 で定義したバスを開く。

    handshake は外す。欠けた ID を見つけるのは呼び出し側の仕事なので、欠けで例外にしない。
    閉じるときは disconnect(disable_torque=False) にする。既定は居ない ID へも書きに行って失敗する。
    """
    motors = {f"id{i}": Motor(i, MODEL, MotorNormMode.DEGREES) for i in range(1, MAX_BUS_ID + 1)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    return bus


def motor_issues(motor_id: int, name: str, regs: dict[str, int | str]) -> list[str]:
    issues: list[str] = []
    voltage = regs.get("Present_Voltage")
    if isinstance(voltage, int) and not VOLTAGE_RAW_MIN <= voltage <= VOLTAGE_RAW_MAX:
        issues.append(f"ID{motor_id} {name}: 電圧 {voltage / 10:.1f}V（期待 11.5-12.6V）")
    temp = regs.get("Present_Temperature")
    if isinstance(temp, int) and temp > TEMP_WARN_C:
        issues.append(f"ID{motor_id} {name}: 温度 {temp}C（>{TEMP_WARN_C}C）")
    status = regs.get("Status")
    if isinstance(status, int) and status != 0:
        issues.append(f"ID{motor_id} {name}: エラーフラグ Status=0x{status:02x}")
    unreadable = [r for r in REGISTERS if isinstance(regs.get(r), str)]
    if unreadable:
        issues.append(f"ID{motor_id} {name}: 読み出し失敗 {', '.join(unreadable)}")
    return issues


def diagnose(port: str) -> tuple[str, frozenset[int], list[str]]:
    """1 ポートを診断し、(バス種別, 応答 ID, 問題のリスト) を返す。"""
    try:
        bus = open_all_ids(port)
    except Exception as e:  # noqa: BLE001  open できない理由ごと報告して続行する
        return "error", frozenset(), [f"{port}: ポート open 失敗: {e}"]

    try:
        found_ids = frozenset(bus.broadcast_ping() or {})
        label, expected = classify_bus(found_ids)

        issues: list[str] = []
        if not found_ids:
            issues.append(f"{port}: 応答なし（12V とデイジーチェーン配線を確認）")
        print(f"\n[{port}] {label}: {len(found_ids)} 台応答")
        print(f"  {'ID':>2} {'name':<28} {'Pos':>5} {'Volt':>5} {'Temp':>4} {'Stat':>4} {'Mode':>4}")
        for i in sorted(found_ids):
            regs: dict[str, int | str] = {}
            for reg in REGISTERS:
                try:
                    regs[reg] = bus.read(reg, f"id{i}", normalize=False)
                except Exception as e:  # noqa: BLE001  1 レジスタの失敗で全体を止めない
                    regs[reg] = f"ERR:{type(e).__name__}"
            name = expected.get(i, f"id{i}(期待外)")
            volt = regs.get("Present_Voltage")
            volt_str = f"{volt / 10:.1f}" if isinstance(volt, int) else "ERR"
            print(
                f"  {i:>2} {name:<28} {regs['Present_Position']!s:>5} {volt_str:>5} "
                f"{regs['Present_Temperature']!s:>4} {regs['Status']!s:>4} {regs['Operating_Mode']!s:>4}"
            )
            issues.extend(motor_issues(i, name, regs))
        for i in sorted(frozenset(expected) - found_ids):
            issues.append(f"{port}: ID{i} {expected[i]} が応答しない")
        return label, found_ids, issues
    finally:
        bus.disconnect(disable_torque=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", action="append", dest="ports", help="診断するポート（複数指定可）")
    args = parser.parse_args()

    ports = args.ports or usb_serial_ports()
    if not ports:
        print("USB シリアルが見つからない")
        return 1

    results = [diagnose(p) for p in ports]
    total = sum(len(found) for _, found, _ in results)
    issues = [msg for _, _, msgs in results for msg in msgs]
    labels = [label for label, _, _ in results]

    print(f"\n応答合計: {total}/17 台")
    for msg in issues:
        print(f"  NG: {msg}")
    ok = total == 17 and labels.count("bus1") == 1 and labels.count("bus2") == 1 and not issues
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
