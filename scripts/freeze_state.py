"""学習データでほとんど動かなかった観測の次元を、ポリシーが無視できるようにしたモデルを、別のディレクトリに作る。

LeRobot は観測を (値 - 平均) / (標準偏差 + 1e-8) で正規化してからポリシーへ入れる。
デモで一度も動かさなかった次元（片手だけの作業での、使わない手の gripper など）は標準偏差がほぼ 0 になり、
評価で少しでも違う値を読むと、正規化した値が数千になる。
指定した次元の標準偏差を 1e6 に書き換えると、正規化した値はいつもほぼ 0 になり、学習のときの入力と同じになる。

Usage:
    python scripts/freeze_state.py <pretrained_model のディレクトリ>
    python scripts/freeze_state.py <pretrained_model のディレクトリ> --name left_arm_gripper.pos
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

KEY = "observation.state"
FROZEN_STD = 1e6


def state_names() -> list[str]:
    """本書の client が出す観測の、関節と速度の名前（データセットの observation.state と同じ順）。"""
    from lerobot_robot_xlerobot_book import XLerobotBookClient, XLerobotBookClientConfig

    features = XLerobotBookClient(XLerobotBookClientConfig(remote_ip="127.0.0.1")).observation_features
    return [k for k, v in features.items() if not isinstance(v, tuple)]  # tuple はカメラの画像


def normalizer_file(model_dir: Path) -> Path:
    files = sorted(model_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if len(files) != 1:
        raise SystemExit(f"{model_dir} に正規化の統計のファイルが 1 つだけあるはずが、{len(files)} 個あった")
    return files[0]


def frozen(stats: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    """指定した次元の標準偏差だけを FROZEN_STD にした統計を返す。元の辞書は変えない。"""
    std = stats[f"{KEY}.std"].copy()
    std[indices] = FROZEN_STD
    return {**stats, f"{KEY}.std": std}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="lerobot-train が書いた pretrained_model のディレクトリ")
    p.add_argument("--name", action="append", default=[], help="無視させる次元の名前。くり返し指定できる")
    p.add_argument("--out", type=Path, help="書き換えたモデルの保存先。既定は <model_dir>_frozen")
    args = p.parse_args()

    src = normalizer_file(args.model_dir)
    stats = load_file(str(src))
    names = state_names()
    std, lo, hi = (stats[f"{KEY}.{s}"] for s in ("std", "min", "max"))
    if len(names) != len(std):
        raise SystemExit(f"統計は {len(std)} 次元で、本書の client の観測（{len(names)} 次元）と合わない")

    if not args.name:
        print(f"{'name':<28}{'std':>12}{'min':>10}{'max':>10}")
        for n, s, a, b in zip(names, std, lo, hi, strict=True):
            print(f"{n:<28}{s:>12.4g}{a:>10.3f}{b:>10.3f}")
        print("学習データで動かさなかった次元を --name で指定すると、その次元を書き換えたモデルを作る")
        return 0

    unknown = [n for n in args.name if n not in names]
    if unknown:
        raise SystemExit(f"知らない名前: {', '.join(unknown)}")
    out = args.out or args.model_dir.with_name(args.model_dir.name + "_frozen")
    if out.exists():
        raise SystemExit(f"{out} はもうある。消すか、--out で別の場所を指定する")
    shutil.copytree(args.model_dir, out)
    indices = [names.index(n) for n in args.name]
    save_file(frozen(stats, indices), str(out / src.name))
    for i in indices:
        print(f"{names[i]}: 標準偏差 {std[i]:.4g} -> {FROZEN_STD:g}")
    print(f"保存先: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
