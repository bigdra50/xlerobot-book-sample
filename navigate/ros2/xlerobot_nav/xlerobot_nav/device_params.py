"""iPad の pocketsensor のパラメータを一度だけ書き換えて終わる。

    ros2 run xlerobot_nav device_params --source ws://<iPad の IP>:8765 --color-rate 5 --color-width 512

書いた値は、アプリがフォアグラウンドにあるあいだ端末に残る。
中継（pocketsensor_ros）は端末のパラメータを変えないので、地図を作るときの launch がこれを先に走らせる。
"""

from __future__ import annotations

import argparse

import pocketsensor as ps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--color-rate", type=float)
    parser.add_argument("--color-width", type=int)
    args, _ = parser.parse_known_args()  # ros2 run が足す --ros-args は読み飛ばす
    config = ps.Config(streams=[ps.Pose()], frame_policy=ps.FramePolicy.ANY)
    with ps.open(args.source, config) as dev:
        if args.color_rate is not None:
            print("color.rate", dev.set_rate(ps.Stream.COLOR, args.color_rate))
        if args.color_width is not None:
            print("color.width", dev.set_color_width(args.color_width))


if __name__ == "__main__":
    main()
