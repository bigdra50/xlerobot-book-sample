#!/usr/bin/env bash
# 残した地図の上で台車に巡回させ、そのあいだの話題を MCAP に記録する。Mac で走らせる。
# 先に map.launch.py を localize:=true で起動しておく。
#
#   patrol.sh [<名前>] [patrol のパラメータ ...]
#   例: patrol.sh demo -p laps:=2 -p extra:='[[-0.03, -0.62]]'
#
# 記録は navigate/gate/runs/ros2/bags/<名前>（MCAP。Lichtblick でそのまま開ける）、
# 結果は navigate/gate/runs/ros2/patrol/<名前>.json に出る。
# Ctrl+C で止めると、Nav2 の目標を取り消してから記録を閉じる。
set -euo pipefail
NAME="${XLEROBOT_NAV_CONTAINER:-xlerobot-nav}"
RUN="${1:-patrol_$(date +%Y%m%d_%H%M%S)}"
shift || true
ARGS=$(printf '%q ' "$@")
TOPICS="/tf /tf_static /robot_description /map /mapPath /cloud_map_dense /scan /plan /odom /wheel/odom /cmd_vel \
/localization_pose /patrol/waypoints /local_costmap/costmap /global_costmap/costmap \
/pocketsensor/color/image/compressed /pocketsensor/color/camera_info"

cleanup() {
  # [x] と [r] は、この bash 自身の引数に一致させないため
  docker exec "${NAME}" bash -lc 'source /ws/env.sh
    pkill -INT -f "[x]lerobot_nav/patrol" || true
    sleep 2
    ros2 service call /navigate_to_pose/_action/cancel_goal action_msgs/srv/CancelGoal "{}" >/dev/null 2>&1 || true
    pid=$(pgrep -f "[r]os2 bag record -s mcap -o /out/bags/'"${RUN}"'" | head -1)
    [ -n "${pid}" ] && kill -INT -- "-${pid}" 2>/dev/null
    for _ in $(seq 20); do [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null && break; sleep 0.5; done' || true
}
trap cleanup EXIT

docker exec "${NAME}" mkdir -p /out/bags /out/patrol
docker exec -d "${NAME}" bash -lc \
  "source /ws/env.sh && exec setsid ros2 bag record -s mcap -o /out/bags/${RUN} ${TOPICS} > /out/bags/${RUN}_record.log 2>&1"
sleep 3
docker exec "${NAME}" bash -lc "source /ws/env.sh && ros2 run xlerobot_nav patrol --ros-args -p out:=/out/patrol/${RUN}.json ${ARGS}"
