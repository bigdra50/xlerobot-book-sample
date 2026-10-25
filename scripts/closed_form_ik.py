"""手先の位置と向きから、SO-101 の 5 関節を閉形式の IK で解く。

本書の plugin が --teleop.ik_backend=closed_form で使うものと同じ so101_chain を呼ぶ。
実機は要らない。install.sh が作る環境で動かす。

Usage:
    python scripts/closed_form_ik.py --selftest
    python scripts/closed_form_ik.py --x 0.25 --y 0.05 --z 0.10 --pitch -30 --roll 0
"""

from __future__ import annotations

import argparse

from lerobot_teleoperator_xlerobot_quest.ik.vendor import so101_chain as chain

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def selftest() -> int:
    """関節角から手先を求め、その手先を解き直して、元の関節角へ戻るかを見る。"""
    worst = 0.0
    for joints in (
        dict.fromkeys(JOINTS, 0.0),
        dict(zip(JOINTS, (20.0, -30.0, 40.0, -15.0, 10.0), strict=True)),
    ):
        tool = chain.forward(joints)["tool"]
        pitch = chain.tool_pitch(joints["shoulder_lift"], joints["elbow_flex"], joints["wrist_flex"])
        solved = chain.inverse(tool, pitch, joints["wrist_roll"]).as_dict()
        worst = max(worst, *(abs(solved[j] - joints[j]) for j in JOINTS))
    print(f"関節角から手先を求めて解き直したときの、関節角のずれの最大: {worst:.6f} 度")
    print("OK" if worst < 1e-6 else "NG")
    return 0 if worst < 1e-6 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--x", type=float, default=0.25, help="腕の根元から前へ（m）")
    parser.add_argument("--y", type=float, default=0.0, help="腕の根元から横へ（m）")
    parser.add_argument("--z", type=float, default=0.10, help="腕の根元から上へ（m）")
    parser.add_argument("--pitch", type=float, default=0.0, help="手先の仰角（度）。上向きが正")
    parser.add_argument("--roll", type=float, default=0.0, help="手先のひねり（度）")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    pose = chain.inverse((args.x, args.y, args.z), args.pitch, args.roll)
    for joint, deg in pose.as_dict().items():
        print(f"  {joint:<14} {deg:8.2f}")
    print(f"  届かなかった距離 {pose.position_error_m * 1000:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
