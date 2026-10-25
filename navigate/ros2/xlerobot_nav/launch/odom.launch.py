"""pocketsensor の中継、ホイールの橋渡し、オドメトリの融合を起動する。

ros2 launch xlerobot_nav odom.launch.py source:=ws://<iPad の IP>:8765 base_host:=<Pi の IP>
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("source", default_value="ws://iphone.local:8765"),
            DeclareLaunchArgument("base_host", default_value="xlerobot-pi.local"),
            DeclareLaunchArgument("yaw_scale", default_value="1.0"),
            Node(
                package="pocketsensor_ros",
                executable="relay",
                name="pocketsensor_relay",
                parameters=[{"source": LaunchConfiguration("source"), "streams": ["pose"]}],
            ),
            Node(
                package="xlerobot_nav",
                executable="base_bridge",
                parameters=[
                    {
                        "host": LaunchConfiguration("base_host"),
                        "yaw_scale": ParameterValue(LaunchConfiguration("yaw_scale"), value_type=float),
                    }
                ],
            ),
            Node(package="xlerobot_nav", executable="odom_fusion"),
        ]
    )
