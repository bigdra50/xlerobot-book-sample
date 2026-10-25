#!/usr/bin/env bash
# コンテナの中で、pocketsensor の SDK を入れ、pocketsensor_msgs、pocketsensor_ros、xlerobot_nav をビルドする。
#   /book    このリポジトリ（読み取り専用）
#   /pocket  pocketsensor（読み取り専用）
set -eo pipefail
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
echo "== pocketsensor の Python SDK を入れる"
rm -rf /tmp/sdk && cp -r /pocket/python /tmp/sdk && pip3 install --quiet --break-system-packages /tmp/sdk
# SDK が numpy 2 を入れるので、apt の python3-opencv（numpy 1 向け）は読み込めなくなる。ArUco の検出に使う OpenCV を pip で入れる
pip3 install --quiet --break-system-packages opencv-contrib-python-headless
echo "== colcon build"
rm -rf /ws && mkdir -p /ws/src
cp -r /pocket/ros2/pocketsensor_msgs /pocket/ros2/pocketsensor_ros /book/navigate/ros2/xlerobot_nav /ws/src/
cd /ws && colcon build --event-handlers console_cohesion- >/tmp/colcon.log 2>&1 || { tail -40 /tmp/colcon.log; exit 1; }
cat > /ws/env.sh <<'ENV'
source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
source /ws/install/setup.bash
# navgate は ROS の外の uv プロジェクト。融合と車輪の計算（numpy と標準ライブラリだけに依存）を、ソースのまま読む
export PYTHONPATH=/book/navigate/gate/src:${PYTHONPATH}
ENV
echo "== ビルドできた"
