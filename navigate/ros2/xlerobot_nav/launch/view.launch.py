"""地図と台車を Lichtblick で見るための foxglove_bridge を起動する。

ros2 launch xlerobot_nav view.launch.py                                   # 走らせているスタックの話題を見る
ros2 launch xlerobot_nav view.launch.py cloud_database:=/out/rtabmap.db   # 同じく、保存した地図の点群を重ねる
ros2 launch xlerobot_nav view.launch.py database_path:=/out/rtabmap.db    # 保存した地図を見る（台車は要らない）

走らせているスタック（map.launch.py、nav.launch.py）と並べて起動すれば、作っている途中の地図、台車の位置、
疑似 LaserScan、costmap、経路、iPad のカラー、巡回の点がそのまま流れる。
保存した地図を見るときは、地図の写しをセンサーの入力なしで rtabmap に読ませ、占有格子と通った道を出させる。
色つきの点群は rtabmap-export で深度から書き出し、/cloud_map_dense に出す（saved_map）。元の地図は変わらない。
残した地図の上で位置を取り直して走らせるときは、cloud_database に同じ地図を渡せば、点群を重ねて見られる。
地図を作っている途中は、データベースが書き換わるので渡さない。

foxglove_bridge はコンテナの port（既定 8765）で待ち受ける。ホストからは up.sh がつないだ 127.0.0.1:8767 で届く。
"""

from __future__ import annotations

import importlib.util
import os

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

WORK_DB = "/tmp/xlerobot_view.db"
# rtabmap-export は --output の名前に _cloud.ply を足す
CLOUD = "/tmp/xlerobot_view_cloud.ply"
# foxglove_bridge が流す話題。rtabmap は使わない話題も多く出し、その型の定義で Lichtblick が警告を出すため絞る
TOPICS = [
    r"^/(tf|tf_static|robot_description)$",
    r"^/(map|cloud_map|cloud_map_dense|mapPath|localization_pose)$",
    r"^/(scan|plan|odom|cmd_vel|diagnostics)$",
    r"^/(local|global)_costmap/costmap$",
    r"^/(pocketsensor|wheel|patrol|xlerobot)/.*",
]


def _map_settings() -> dict:
    """地図を作ったときの rtabmap の設定を map.launch.py から読む。占有格子の作り方をそろえるため。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.launch.py")
    spec = importlib.util.spec_from_file_location("map_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RTABMAP


def _saved(context: LaunchContext) -> list:
    saved = LaunchConfiguration("database_path").perform(context)
    db = saved or LaunchConfiguration("cloud_database").perform(context)
    if not db:
        return []
    voxel = LaunchConfiguration("cloud_voxel").perform(context)
    max_range = LaunchConfiguration("cloud_max_range").perform(context)
    export = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            # 半径 5 cm に 5 点ない点は、ガラス越しや反射の深度の外れ値として捨てる
            (
                f"cp {db} {WORK_DB} && rm -f {CLOUD} && rtabmap-export --cloud --decimation 2 --voxel {voxel} "
                f"--max_range {max_range} --noise_radius 0.05 --noise_k 5 --output xlerobot_view --output_dir /tmp "
                f"{WORK_DB} > /tmp/xlerobot_view_export.log"
            ),
        ],
        output="screen",
    )

    def after_export(event, _context) -> list:
        saved_map = Node(
            package="xlerobot_nav",
            executable="saved_map",
            parameters=[
                {
                    "cloud": CLOUD if event.returncode == 0 else "",
                    # 走らせているスタックに重ねるときは、点群だけを出す
                    "publish_map_service": "/rtabmap/publish_map" if saved else "",
                }
            ],
            output="screen",
        )
        if not saved:
            return [saved_map]
        rtabmap = Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            parameters=[
                {
                    **_map_settings(),
                    "database_path": WORK_DB,
                    # 地図を増やさず、残した地図の全体を読む。センサーの入力は受けない
                    "Mem/IncrementalMemory": "false",
                    "Mem/InitWMWithAllNodes": "true",
                    "subscribe_rgbd": False,
                    "subscribe_depth": False,
                    "subscribe_rgb": False,
                    "subscribe_scan": False,
                    "publish_tf": False,
                    "frame_id": "base_link",
                    "map_frame_id": "map",
                }
            ],
            # rtabmap の粗い点群（占有格子のマスの点）は、書き出した点群と重ならないよう別の名前へ逃がす。
            # 通った道は、saved_map が受けて TRANSIENT_LOCAL で /mapPath へ出し直す
            remappings=[("cloud_map", "cloud_map_grid"), ("mapPath", "mapPath_rtabmap")],
            # 入力が無いので、5 秒ごとの「データが来ない」警告を出させない
            ros_arguments=["--log-level", "rtabmap:=error"],
        )
        return [rtabmap, saved_map]

    return [export, RegisterEventHandler(OnProcessExit(target_action=export, on_exit=after_export))]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="8765"),
            DeclareLaunchArgument("database_path", default_value=""),
            DeclareLaunchArgument("cloud_database", default_value=""),
            # 書き出す点群の間隔と、深度から使う距離の上限
            DeclareLaunchArgument("cloud_voxel", default_value="0.01"),
            DeclareLaunchArgument("cloud_max_range", default_value="3.5"),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                parameters=[
                    {
                        "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
                        "address": "0.0.0.0",
                        # 保存した地図の点群は 1 通で約 8 MB ある。既定の 10 MB では、画像と重なると落とす
                        "send_buffer_limit": 100_000_000,
                        "topic_whitelist": TOPICS,
                    }
                ],
            ),
            OpaqueFunction(function=_saved),
        ]
    )
