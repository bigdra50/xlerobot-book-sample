#!/usr/bin/env python
"""本書の実行環境が組めているかを、実機なしで確かめる。

lerobot が v0.6.1 で入っていること、feetech、core-scripts、training の extras が
あること、plugin が lerobot-* CLI から見える名前で入っていること、そして plugin が
XLeRobot を 17 軸（bus1 が 8 台、bus2 が 9 台）として登録していることを見る。

    python verify_stack.py
    python verify_stack.py --features    # 17 軸の割り当てを実装から引く

XLerobot のコンストラクタはシリアルポートを開かないので、実機は要らない。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from importlib.metadata import distributions

EXPECTED_LEROBOT = "0.6.1"


def _report(label: str, ok: bool, detail: str = "") -> bool:
    mark = "  ok  " if ok else " FAIL "
    print(f"[{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check() -> bool:
    ok = True
    print("== XLeRobot 実行環境の確認 ==")
    print(f"python: {sys.executable}")

    try:
        import lerobot

        ok &= _report(
            "lerobot import", lerobot.__version__ == EXPECTED_LEROBOT, f"version {lerobot.__version__}"
        )
    except ImportError as e:
        return _report("lerobot import", False, str(e))

    for extra, module in (
        ("feetech", "lerobot.motors.feetech"),
        ("core-scripts", "lerobot.datasets"),
        ("training", "accelerate"),
    ):
        found = _importable(module)
        ok &= _report(f"lerobot[{extra}]", found, "" if found else f"extras に {extra} を付ける")

    names = {d.metadata.get("Name", "").replace("-", "_") for d in distributions()}
    for dist, how in (
        ("lerobot_robot_xlerobot", "install.sh で公式リポジトリの software/plugins から入れる"),
        ("lerobot_robot_xlerobot_book", "uv pip install -e plugins/lerobot_robot_xlerobot_book"),
    ):
        installed = dist in names
        ok &= _report(f"{dist} が install 済み", installed, "" if installed else how)
        if not installed:
            return False

    try:
        importlib.import_module("lerobot_robot_xlerobot_book")
        plugin = importlib.import_module("lerobot_robot_xlerobot")
    except ImportError as e:
        return ok & _report("plugin の import", False, str(e))

    from lerobot.robots.config import RobotConfig

    types = ("xlerobot", "xlerobot_book_client")
    registered = all(RobotConfig.get_choice_class(n) is not None for n in types)
    ok &= _report(" / ".join(types) + " が robot type に登録済み", registered)

    robot = plugin.XLerobot(plugin.XLerobotConfig(id="verify"))
    n_action = len(robot.action_features)
    ok &= _report("17 次元の action_features", n_action == 17, f"{n_action} 次元")
    n1, n2 = len(robot.bus1.motors), len(robot.bus2.motors)
    ok &= _report("17 モーター（bus1=8 / bus2=9）", (n1, n2) == (8, 9), f"{n1}+{n2}")

    print()
    print("すべてOK。実機を繋いだキャリブレーションへ進める。" if ok else "FAIL がある。上の行から疑う。")
    return ok


def features() -> None:
    plugin = importlib.import_module("lerobot_robot_xlerobot")
    robot = plugin.XLerobot(plugin.XLerobotConfig(id="verify"))
    for bus_name, bus in (("bus1", robot.bus1), ("bus2", robot.bus2)):
        for name, motor in bus.motors.items():
            print(f"{bus_name}  ID{motor.id}  {name:24s} {motor.norm_mode.name}")
    print()
    print(f"action_features ({len(robot.action_features)}):")
    print("  " + " ".join(robot.action_features))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", action="store_true", help="17 軸の割り当てを実装から引いて表示する")
    args = p.parse_args()
    if args.features:
        features()
        return 0
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
