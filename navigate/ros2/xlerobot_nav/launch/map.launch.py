"""地図を作り、地図の上で Nav2 を動かす。nav.launch.py（センサー、オドメトリ、Nav2）に、iPad のカラーと rtabmap を足す。

ros2 launch xlerobot_nav map.launch.py source:=ws://<iPad の IP>:8765 base_host:=<Pi の IP> [new:=true | localize:=true | append:=true]

地図は database_path（既定はコンテナの /out/rtabmap.db）に残り、/map に占有格子を、map -> odom に地図の上の位置を出す。
new:=true で、前の地図を消して作り直す。
localize:=true では地図を増やさず、残した地図の上で位置だけを取り直す。
append:=true では、家具を動かしたあとの様子を、残した地図へ新しいセッションとして撮り足す。
Nav2 は地図を使い、目標を map の座標で受ける。
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetLaunchConfiguration,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# rtabmap は RGB と深度の解像度が整数倍であることを求める。深度は 256x192 なので、カラーを 2 倍の 512x384 にする。
# 地図の更新は 1 Hz で足りるので、カラーは 5 Hz にして WiFi の帯域を深度へ回す
COLOR_WIDTH = "512"
COLOR_RATE = "5"

RTABMAP = {
    "Rtabmap/DetectionRate": "1",
    # 狭い部屋で、少しの移動と旋回でも節点を足す
    "RGBD/LinearUpdate": "0.05",
    "RGBD/AngularUpdate": "0.05",
    # 床を走る台車なので、姿勢は x、y、yaw だけで最適化する
    "Reg/Force3DoF": "true",
    "Optimizer/GravitySigma": "0",
    "Kp/MaxFeatures": "400",
    "Vis/MinInliers": "12",
    # 占有格子は深度から作る。床の高さの帯と障害物の帯は、疑似 LaserScan とそろえる
    "Grid/Sensor": "1",
    "Grid/3D": "false",
    "Grid/CellSize": "0.05",
    "Grid/RangeMax": "3.0",
    "Grid/RayTracing": "true",
    "Grid/NormalsSegmentation": "false",
    "Grid/MaxGroundHeight": "0.05",
    "Grid/MaxObstacleHeight": "1.25",
}

# 位置だけを取り直すときの、取り直しに要る一致の点の下限。地図を作るとき（Vis/MinInliers）より厳しくする。
# 机の前の視野（天板の裏、柱、奥の壁）での取り直しは一致 12〜58 点で、地図の上の位置を最大 15 cm 跳ばせた。
# 通路で一致 50〜70 点の取り直しは、地図の節点どうしの食い違い（最大 8 cm）を待機位置へ持ち込んだ
# 撮り足すときも、新しいセッションを残した地図へつなぐ条件に、同じ下限を使う
LOCALIZE_MIN_INLIERS = "70"

# rgbd_sync と rtabmap のノードの設定
RGBD_SYNC = {
    "image_transport": "compressed",
    "approx_sync": True,
    "approx_sync_max_interval": 0.02,
    # 中継は画像を best effort で出す
    "qos": 2,
    "qos_camera_info": 2,
    "sync_queue_size": 30,
}
RGBD_SYNC_REMAPS = [
    ("rgb/image", "/pocketsensor/color/image"),
    ("rgb/camera_info", "/pocketsensor/color/camera_info"),
    ("depth/image", "/pocketsensor/depth/image"),
    ("rgbd_image", "/rgbd_image"),
]
RTABMAP_NODE = {
    "frame_id": "base_link",
    "odom_frame_id": "odom",
    "map_frame_id": "map",
    "publish_tf": True,
    "subscribe_rgbd": True,
    "subscribe_depth": False,
    "subscribe_rgb": False,
    "approx_sync": False,
    # 融合した odom -> base_link は最長 0.3 秒遅れる
    "wait_for_transform": 0.5,
    "odom_tf_linear_variance": 0.0005,
    "odom_tf_angular_variance": 0.0005,
}


def _str(expression: list) -> ParameterValue:
    # launch は "true" のような文字列を bool に直すが、rtabmap の設定は文字列で受ける
    return ParameterValue(PythonExpression(expression), value_type=str)


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare("xlerobot_nav")
    device_params = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "xlerobot_nav",
            "device_params",
            "--source",
            LaunchConfiguration("source"),
            "--color-rate",
            COLOR_RATE,
            "--color-width",
            COLOR_WIDTH,
        ],
        output="screen",
    )
    # カラーの解像度を変えてから流し始める。先に流すと、960x720 のフレームを rtabmap が受けて整数倍の条件に反する
    after_params = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([share, "launch", "nav.launch.py"]))
        ),
        # カラーと深度は大きいので、深度の処理と同じプロセスへ載せて、プロセス内通信で渡す
        LoadComposableNodes(
            target_container="sensor_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="rtabmap_sync",
                    plugin="rtabmap_sync::RGBDSync",
                    name="rgbd_sync",
                    parameters=[RGBD_SYNC],
                    remappings=RGBD_SYNC_REMAPS,
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
                ComposableNode(
                    package="rtabmap_slam",
                    plugin="rtabmap_slam::CoreWrapper",
                    name="rtabmap",
                    parameters=[
                        {
                            **RTABMAP_NODE,
                            "database_path": LaunchConfiguration("database_path"),
                            "delete_db_on_start": ParameterValue(LaunchConfiguration("new"), value_type=bool),
                            **RTABMAP,
                            # 位置だけを取り直すときは地図を増やさず、残した地図の全体を最初から照合の相手にする
                            "Mem/IncrementalMemory": _str(
                                ["'false' if '", LaunchConfiguration("localize"), "' == 'true' else 'true'"]
                            ),
                            "Mem/InitWMWithAllNodes": _str(
                                [
                                    "'true' if 'true' in ('",
                                    LaunchConfiguration("localize"),
                                    "', '",
                                    LaunchConfiguration("append"),
                                    "') else 'false'",
                                ]
                            ),
                            "Vis/MinInliers": _str(
                                [
                                    f"'{LOCALIZE_MIN_INLIERS}' if 'true' in ('",
                                    LaunchConfiguration("localize"),
                                    "', '",
                                    LaunchConfiguration("append"),
                                    f"') else '{RTABMAP['Vis/MinInliers']}'",
                                ]
                            ),
                        }
                    ],
                    remappings=[("rgbd_image", "/rgbd_image")],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
        ),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("source", default_value="ws://iphone.local:8765"),
            DeclareLaunchArgument("base_host", default_value="xlerobot-pi.local"),
            DeclareLaunchArgument("database_path", default_value="/out/rtabmap.db"),
            DeclareLaunchArgument("new", default_value="false"),
            DeclareLaunchArgument("localize", default_value="false"),
            DeclareLaunchArgument("append", default_value="false"),
            SetLaunchConfiguration("color", "true"),
            SetLaunchConfiguration("use_map", "true"),
            device_params,
            RegisterEventHandler(OnProcessExit(target_action=device_params, on_exit=after_params)),
        ]
    )
