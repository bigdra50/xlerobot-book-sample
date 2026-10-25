"""センサーとオドメトリ（bringup.launch.py）に、Nav2 を足して起動する。

ros2 launch xlerobot_nav nav.launch.py source:=ws://<iPad の IP>:8765 base_host:=<Pi の IP> [use_map:=true]

use_map:=false（既定）では地図を使わず、目標を odom の座標で /navigate_to_pose へ送る。odom の原点は起動したときの台車の位置になる。
use_map:=true では、global costmap が rtabmap の /map を土台にし、目標を map の座標で送る（map.launch.py が使う）。
Nav2 の標準の navigation_launch.py は使わない。route_server やドッキングなど、ここで使わないサーバーまで起動するため。
"""

import tempfile

import yaml
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

NAV_NODES = [
    ("nav2_controller", "controller_server", True),
    ("nav2_planner", "planner_server", False),
    ("nav2_behaviors", "behavior_server", True),
    ("nav2_velocity_smoother", "velocity_smoother", True),
    ("nav2_collision_monitor", "collision_monitor", False),
    ("nav2_bt_navigator", "bt_navigator", False),
]


def map_params(params: dict) -> dict:
    """地図なしの設定から、global costmap と目標を map の座標で扱う設定を作る。ほかは同じにする。"""
    params["bt_navigator"]["ros__parameters"]["global_frame"] = "map"
    params["behavior_server"]["ros__parameters"]["global_frame"] = "map"
    g = params["global_costmap"]["global_costmap"]["ros__parameters"]
    for key in ("width", "height"):
        g.pop(key, None)  # 大きさは地図で決まる
    g.update(
        global_frame="map",
        rolling_window=False,
        # 地図の未知の場所は未知のまま残し、planner の allow_unknown で通れるようにする
        track_unknown_space=True,
        plugins=["static_layer", "obstacle_layer", "inflation_layer"],
        static_layer={
            "plugin": "nav2_costmap_2d::StaticLayer",
            "map_topic": "/map",
            "map_subscribe_transient_local": True,
        },
    )
    return params


def _nav_nodes(context: LaunchContext) -> list:
    share = FindPackageShare("xlerobot_nav").perform(context)
    params = f"{share}/config/nav2.yaml"
    if LaunchConfiguration("use_map").perform(context).lower() == "true":
        with open(params) as f:
            rewritten = map_params(yaml.safe_load(f))
        # Nav2 のノードは起動してからこのファイルを読むので、launch が終わるまで消さない
        with tempfile.NamedTemporaryFile("w", prefix="nav2_map_", suffix=".yaml", delete=False) as out:
            yaml.safe_dump(rewritten, out)
        params = out.name
    bt = f"{share}/config/navigate_to_pose.xml"
    # 指令の流れ: controller と behavior -> cmd_vel_nav -> velocity_smoother -> cmd_vel_smoothed
    #   -> collision_monitor -> cmd_vel -> base_bridge
    return [
        Node(
            package=pkg,
            executable=exe,
            name=exe,
            output="screen",
            parameters=[params, {"default_nav_to_pose_bt_xml": bt}] if exe == "bt_navigator" else [params],
            remappings=[("cmd_vel", "cmd_vel_nav")] if to_nav else [],
        )
        for pkg, exe, to_nav in NAV_NODES
    ]


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("xlerobot_nav")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_map", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([share, "launch", "bringup.launch.py"]))
            ),
            OpaqueFunction(function=_nav_nodes),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"autostart": True, "node_names": [exe for _, exe, _ in NAV_NODES], "bond_timeout": 10.0}
                ],
            ),
        ]
    )
