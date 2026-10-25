#!/usr/bin/env bash
# ROS 2 のコンテナを作り、pocketsensor の中継と xlerobot_nav をビルドする。コンテナは残し、docker exec で使う。
#
#   XLEROBOT_NAV_BASE       Dockerfile の土台にする ROS 2 Jazzy のイメージ（既定 ros:jazzy-ros-base）
#   XLEROBOT_NAV_CONTAINER  コンテナの名前（既定 xlerobot-nav）
#   POCKETSENSOR_DIR        pocketsensor のリポジトリ（既定は、このリポジトリと並んだ ../pocketsensor）
#   XLEROBOT_NAV_VIEW_PORT  Lichtblick からつなぐ foxglove_bridge の、ホストのポート（既定 8767。view.sh が使う）
#   ROS_DOMAIN_ID           コンテナの中のスタックの ROS_DOMAIN_ID（既定 0）
#
# 記録と結果は navigate/gate/runs/ros2/ に出る（コンテナの /out）。
set -euo pipefail
BOOK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POCKET="${POCKETSENSOR_DIR:-$(cd "${BOOK}/../pocketsensor" && pwd)}"
BASE="${XLEROBOT_NAV_BASE:-ros:jazzy-ros-base}"
IMAGE=xlerobot-nav:jazzy
NAME="${XLEROBOT_NAV_CONTAINER:-xlerobot-nav}"
OUT="${BOOK}/navigate/gate/runs/ros2"
VIEW_PORT="${XLEROBOT_NAV_VIEW_PORT:-8767}"
mkdir -p "${OUT}"
# Nav2 と深度の変換を足したイメージ。2 回目からは docker のキャッシュで済む
docker build -q -t "${IMAGE}" --build-arg BASE="${BASE}" "${BOOK}/navigate/ros2/docker" >/dev/null
docker rm -f "${NAME}" >/dev/null 2>&1 || true
# foxglove_bridge（コンテナの 8765 番）は、このマシンからだけ届くようにする
docker run -d --name "${NAME}" -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" -p "127.0.0.1:${VIEW_PORT}:8765" -v "${BOOK}:/book:ro" -v "${POCKET}:/pocket:ro" -v "${OUT}:/out" \
  --entrypoint sleep "${IMAGE}" infinity >/dev/null
docker exec "${NAME}" bash -lc /book/navigate/ros2/docker/build_in_container.sh
cat <<MSG
起動（地図を作る。残した地図の上で走らせるときは new:=true を localize:=true にする）:
  docker exec -d ${NAME} bash -lc 'source /ws/env.sh && ros2 launch xlerobot_nav map.launch.py source:=ws://<iPad の IP>:8765 base_host:=<Pi の IP> new:=true > /out/launch.log 2>&1'
地図と台車を見る:
  $(dirname "$0")/view.sh saved | live
片付け:
  docker rm -f ${NAME}
MSG
