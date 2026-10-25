#!/usr/bin/env bash
# 地図と台車を Lichtblick で見る。Mac で走らせる。コンテナは up.sh で作ったものを使う。
#
#   view.sh saved [<コンテナの中の地図のパス>]   保存した地図を見る（既定 /out/rtabmap.db）。台車は要らない
#   view.sh live [<地図のパス>]                  走らせているスタック（map.launch.py、nav.launch.py）の話題を見る。
#                                               地図を渡すと、その地図の色つきの点群を重ねる（位置の取り直しのモードで使う）
#   view.sh stop                                表示を止める
#
#   XLEROBOT_NAV_CONTAINER   コンテナの名前（既定 xlerobot-nav）
#   XLEROBOT_NAV_VIEW_PORT   foxglove_bridge をつないだホストのポート（既定 8767。up.sh と同じ値にする）
#   XLEROBOT_LICHTBLICK_PORT Lichtblick の Web 版のポート（既定 8081）
#   ROS_DOMAIN_ID            見るスタックの ROS_DOMAIN_ID（既定はコンテナの値）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYOUT_DIR="$(cd "${HERE}/../lichtblick" && pwd)"
NAME="${XLEROBOT_NAV_CONTAINER:-xlerobot-nav}"
WS_PORT="${XLEROBOT_NAV_VIEW_PORT:-8767}"
UI_PORT="${XLEROBOT_LICHTBLICK_PORT:-8081}"
UI=xlerobot-lichtblick
MODE="${1:-live}"

stop_view() {
  # launch は setsid で起動しているので、プロセスグループごと止める
  docker exec "${NAME}" bash -c '
    # [l] は、この bash 自身の引数に一致させないため
    pid=$(pgrep -f "[l]aunch xlerobot_nav view.launch.py" | head -1)
    [ -z "${pid}" ] && exit 0
    kill -INT -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 20); do kill -0 "${pid}" 2>/dev/null || exit 0; sleep 0.5; done
    kill -KILL -- "-${pid}" 2>/dev/null || true'
}

case "${MODE}" in
  saved) ARGS="database_path:=${2:-/out/rtabmap.db}" ;;
  live) ARGS="${2:+cloud_database:=$2}" ;;
  stop)
    stop_view
    echo "止めた。Lichtblick も片付けるときは docker rm -f ${UI}"
    exit 0
    ;;
  *)
    echo "usage: view.sh saved [<db>] | live [<db>] | stop" >&2
    exit 2
    ;;
esac

if ! docker ps --format '{{.Ports}}' --filter "name=^${NAME}$" | grep -q ":${WS_PORT}->8765"; then
  echo "${NAME} が ${WS_PORT} 番を公開していない。up.sh でコンテナを作り直す" >&2
  exit 1
fi
# レイアウトはディレクトリごとつなぐ。Docker Desktop はファイル 1 つだけをつなぐと、書き換えたあとの長さを反映しない
mounted=$(docker inspect "${UI}" --format '{{range .Mounts}}{{.Source}}{{end}}' 2>/dev/null || true)
if [ "${mounted}" != "${LAYOUT_DIR}" ] || ! docker ps --format '{{.Names}}' | grep -qx "${UI}"; then
  docker rm -f "${UI}" >/dev/null 2>&1 || true
  docker run -d --name "${UI}" -p "127.0.0.1:${UI_PORT}:8080" -v "${LAYOUT_DIR}:/src/xlerobot:ro" \
    ghcr.io/lichtblick-suite/lichtblick >/dev/null
fi
stop_view
docker exec -d ${ROS_DOMAIN_ID:+-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"} "${NAME}" bash -lc \
  "source /ws/env.sh && exec setsid ros2 launch xlerobot_nav view.launch.py ${ARGS} > /out/view.log 2>&1"
# Lichtblick は、一度読んだ layoutUrl のレイアウトを覚えていて、同じ URL では書き換えを読み直さない。中身から作った印を付ける
LAYOUT_VER=$(shasum "${LAYOUT_DIR}/layout.json" | cut -c1-8)
echo "http://localhost:${UI_PORT}/?ds=foxglove-websocket&ds.url=ws://localhost:${WS_PORT}&layoutUrl=http://localhost:${UI_PORT}/xlerobot/layout.json%3Fv%3D${LAYOUT_VER}"
echo "ログは navigate/gate/runs/ros2/view.log。止めるときは view.sh stop"
