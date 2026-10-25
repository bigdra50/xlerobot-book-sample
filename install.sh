#!/usr/bin/env bash
# 本書の実行環境を専用 venv に作る。
#
#   ./install.sh [インストール先]             既定は ~/xlerobot-book-env
#
# 入れるのは lerobot v0.6.1（PyPI）、公式 XLeRobot の plugin、本リポジトリの plugin。
# 公式の plugin は PyPI に無いので、公式リポジトリを <インストール先>/XLeRobot へ clone して入れる。
set -euo pipefail

DEST=""
for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,4p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "不明なオプション: $arg" >&2; exit 2 ;;
    *)
      [ -z "$DEST" ] || { echo "インストール先は1つだけ: $arg" >&2; exit 2; }
      DEST="$arg" ;;
  esac
done
DEST="${DEST:-$HOME/xlerobot-book-env}"
LEROBOT_VERSION="0.6.1"
# lerobot 0.6.1 は Requires-Python >=3.12。3.14 では draccus（CLI 引数パーサ）が動かない
PYTHON_VERSION="3.12"
# feetech      = STS3215 のシリアルサーボ SDK
# core-scripts = dataset + hardware + viz。0.6.1 で基本 install から分離された。
#                dataset が無いと lerobot-record / lerobot-rollout が import の時点で落ちる
# training     = accelerate + wandb。lerobot-train が accelerate を import する。
#                クライアントの Mac でも学習できるよう、最初から入れる
LEROBOT_EXTRAS="feetech,core-scripts,training"
# 公式 XLeRobot の main（2026-10-06、PR #151 で plugin になったコミット）
XLEROBOT_COMMIT="b017b5e6354bd9f61f4247a920c72622ca0aade0"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null || { echo "uv が必要: https://docs.astral.sh/uv/"; exit 1; }

mkdir -p "$DEST"
[ -d "$DEST/venv" ] || uv venv --python "$PYTHON_VERSION" "$DEST/venv"
export VIRTUAL_ENV="$DEST/venv"

# 公式リポジトリはコミットを固定して浅く取る（--depth 1 は branch しか指せないので fetch で指す）
XLEROBOT="$DEST/XLeRobot"
[ -d "$XLEROBOT/.git" ] || git init -q "$XLEROBOT"
if [ "$(git -C "$XLEROBOT" rev-parse -q --verify HEAD || true)" != "$XLEROBOT_COMMIT" ]; then
  git -C "$XLEROBOT" fetch -q --depth 1 https://github.com/Vector-Wangel/XLeRobot.git "$XLEROBOT_COMMIT"
  git -C "$XLEROBOT" checkout -q FETCH_HEAD
fi
OFFICIAL="$XLEROBOT/software/plugins"

uv pip install "lerobot[$LEROBOT_EXTRAS]==$LEROBOT_VERSION"
# 公式の VR の plugin は xlerobot_model に依存し、それも PyPI に無いので、先に入れる
uv pip install -e "$OFFICIAL/xlerobot_model" -e "$OFFICIAL/lerobot_robot_xlerobot"
uv pip install -e "$REPO/plugins/lerobot_robot_xlerobot_book"
# 公式の VR の Teleoperator（--teleop.type=xlerobot_vr_official で試す）
uv pip install -e "$OFFICIAL/lerobot_teleoperator_xlerobot_vr"
# pyroki は PyPI に無く git から入る。jax を含めて数分かかる
uv pip install -e "$REPO/plugins/lerobot_teleoperator_xlerobot_quest[vr,xlevr]"
# README のテストを、この venv の python -m pytest で走らせる
uv pip install pytest

echo
echo "完了: $DEST"
echo "  python : $DEST/venv/bin/python"
echo "  lerobot: $LEROBOT_VERSION ($LEROBOT_EXTRAS)"
echo
echo "次: $DEST/venv/bin/python $REPO/scripts/verify_stack.py"
