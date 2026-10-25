"""台車のセンサーとオドメトリをまとめて起動する。

ros2 launch xlerobot_nav bringup.launch.py source:=ws://<iPad の IP>:8765 base_host:=<Pi の IP>

iPad の深度から疑似 LaserScan（/scan）を作り、車輪と ARKit を融合した /odom と odom -> base_link を出す。
base_link から iPad への変換は urdf/xlerobot.urdf.xacro の取り付けの値で決まる。
"""

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

DEPTH = "/pocketsensor/depth"


def _float(name: str) -> ParameterValue:
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _relay(context: LaunchContext) -> list[Node]:
    streams = ["pose", "depth"]
    if LaunchConfiguration("color").perform(context).lower() == "true":
        streams.append("color")
    # 取り付けの変換は URDF が出す。中継の TF を止めないと、pocketsensor_link の親が 2 つになる
    return [
        Node(
            package="pocketsensor_ros",
            executable="relay",
            name="pocketsensor_relay",
            parameters=[{"source": LaunchConfiguration("source"), "streams": streams, "publish_tf": False}],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    urdf = PathJoinSubstitution([FindPackageShare("xlerobot_nav"), "urdf", "xlerobot.urdf.xacro"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("source", default_value="ws://iphone.local:8765"),
            DeclareLaunchArgument("base_host", default_value="xlerobot-pi.local"),
            DeclareLaunchArgument("yaw_scale", default_value="1.0"),
            # 地図を作るときだけカラーを流す（map.launch.py が true にする）
            DeclareLaunchArgument("color", default_value="false"),
            # 床の点を拾わない下限と、台車の背丈（head まで約 1.2 m）
            DeclareLaunchArgument("scan_min_height", default_value="0.05"),
            DeclareLaunchArgument("scan_max_height", default_value="1.25"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": ParameterValue(Command(["xacro ", urdf]), value_type=str)}],
            ),
            OpaqueFunction(function=_relay),
            # 深度から /scan までを 1 つのプロセスに入れ、プロセス内通信でつなぐ。
            # 点群は 1 通が約 600 KB あり、DDS を通すと共有メモリの区画を超えて、Nav2 が動くと 3 割ほど取りこぼした
            ComposableNodeContainer(
                name="sensor_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container",
                composable_node_descriptions=[
                    # 深度は PNG で届く。depth_image_proc は sensor_msgs/Image を読むので戻す
                    ComposableNode(
                        package="image_transport",
                        plugin="image_transport::Republisher",
                        name="depth_republish",
                        parameters=[
                            {
                                "in_transport": "compressedDepth",
                                "out_transport": "raw",
                                f"qos_overrides.{DEPTH}/image/compressedDepth.subscription.reliability": "best_effort",
                            }
                        ],
                        remappings=[
                            ("in/compressedDepth", f"{DEPTH}/image/compressedDepth"),
                            ("out", f"{DEPTH}/image"),
                        ],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                    ComposableNode(
                        package="depth_image_proc",
                        plugin="depth_image_proc::PointCloudXyzNode",
                        name="depth_points",
                        # 深度は WiFi で最大 0.8 秒ぶんまとめて届く。既定の 5 では、画像と camera_info の組み合わせ待ちがあふれる
                        parameters=[{"queue_size": 30}],
                        remappings=[
                            ("image_rect", f"{DEPTH}/image"),
                            ("camera_info", f"{DEPTH}/camera_info"),
                            ("points", f"{DEPTH}/points"),
                        ],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                    # 3D の点を base_link へ移してから高さの帯で切る。取り付けの向きや傾きが変わっても同じ処理で済む
                    ComposableNode(
                        package="pointcloud_to_laserscan",
                        plugin="pointcloud_to_laserscan::PointCloudToLaserScanNode",
                        name="depth_to_scan",
                        parameters=[
                            {
                                "target_frame": "base_link",
                                "transform_tolerance": 0.05,
                                "min_height": _float("scan_min_height"),
                                "max_height": _float("scan_max_height"),
                                # iPad の水平の視野は約 61°。外側は inf になり、costmap は inf を無視する
                                "angle_min": -0.7,
                                "angle_max": 0.7,
                                "angle_increment": 0.0087,
                                "scan_time": 1.0 / 15.0,
                                "range_min": 0.2,
                                "range_max": 4.0,
                                "use_inf": True,
                            }
                        ],
                        remappings=[("cloud_in", f"{DEPTH}/points"), ("scan", "/scan")],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                ],
            ),
            Node(
                package="xlerobot_nav",
                executable="base_bridge",
                parameters=[{"host": LaunchConfiguration("base_host"), "yaw_scale": _float("yaw_scale")}],
            ),
            Node(package="xlerobot_nav", executable="odom_fusion"),
        ]
    )
