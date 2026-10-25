"""度とモーターの正規化値の変換。lerobot 0.6.1 の MotorsBus._normalize / _unnormalize と同じ式。

IK は度で解き、robot 側（公式 host は use_degrees=False）は -100..100 を受け取る。
robot 側で別の式に通されると二重変換になるので、ここでは lerobot と同じ式を再現するだけにし、
homing_offset は使わない（lerobot はそれをモーター側の Homing_Offset レジスタで処理する）。
"""

from __future__ import annotations

from lerobot.motors import MotorCalibration

from .contracts import MOTOR_NAMES, JointLimits

# STS3215 の分解能。lerobot の model_resolution_table["sts3215"] と同じ値で、
# lerobot の DEGREES 変換は (resolution - 1) で割る
STS3215_RESOLUTION = 4096
_MAX_RES = STS3215_RESOLUTION - 1


def _mid(cal: MotorCalibration) -> float:
    return (cal.range_min + cal.range_max) / 2


def _check_range(cal: MotorCalibration) -> None:
    if cal.range_max == cal.range_min:
        raise ValueError(f"range_min と range_max が同じ（{cal.range_min}）。キャリブレーションが壊れている")


def raw_to_deg(raw: float, cal: MotorCalibration) -> float:
    return (raw - _mid(cal)) * 360 / _MAX_RES


def deg_to_raw(deg: float, cal: MotorCalibration) -> float:
    # lerobot は int() で切り捨てるが、ここでは丸めない。robot 側へ渡す前に丸めると往復で 1 tick ずれる
    return deg * _MAX_RES / 360 + _mid(cal)


def raw_to_normalized(raw: float, cal: MotorCalibration) -> float:
    """RANGE_M100_100。Feetech は apply_drive_mode=True なので drive_mode で符号を反転する。"""
    _check_range(cal)
    bounded = min(cal.range_max, max(cal.range_min, raw))
    norm = ((bounded - cal.range_min) / (cal.range_max - cal.range_min)) * 200 - 100
    return -norm if cal.drive_mode else norm


def normalized_to_raw(norm: float, cal: MotorCalibration) -> float:
    _check_range(cal)
    val = -norm if cal.drive_mode else norm
    bounded = min(100.0, max(-100.0, val))
    return ((bounded + 100) / 200) * (cal.range_max - cal.range_min) + cal.range_min


def deg_to_normalized(deg: float, cal: MotorCalibration) -> float:
    return raw_to_normalized(deg_to_raw(deg, cal), cal)


def normalized_to_deg(norm: float, cal: MotorCalibration) -> float:
    return raw_to_deg(normalized_to_raw(norm, cal), cal)


def limits_from_calibration(cal: MotorCalibration) -> JointLimits:
    """range_min / range_max を度に換算する。全周（0..4095）なら ±180 になる。"""
    return JointLimits(lower_deg=raw_to_deg(cal.range_min, cal), upper_deg=raw_to_deg(cal.range_max, cal))


def to_robot_units(
    deg_by_motor: dict[str, float], calibration: dict[str, MotorCalibration]
) -> dict[str, float]:
    """関節の度を robot 側の正規化値へ。キーは入力のまま（".pos" は呼び出し側が付ける）。

    gripper はトリガーから直接 0..100 で決めるので素通し。
    MOTOR_NAMES に無いキーは変換先が決められないので KeyError にする。
    素通しにすると度がそのまま正規化値として送られ、気づきにくい。
    """
    out: dict[str, float] = {}
    for name, value in deg_by_motor.items():
        if name not in MOTOR_NAMES:
            raise KeyError(f"{name!r} は MOTOR_NAMES に無い")
        if name.endswith("gripper"):
            out[name] = value
        else:
            out[name] = deg_to_normalized(value, calibration[name])
    return out


def from_robot_units(
    norm_by_motor: dict[str, float], calibration: dict[str, MotorCalibration]
) -> dict[str, float]:
    """to_robot_units の逆。robot 側の正規化値を関節の度へ。gripper は素通し。"""
    return {
        name: value if name.endswith("gripper") else normalized_to_deg(value, calibration[name])
        for name, value in norm_by_motor.items()
    }
