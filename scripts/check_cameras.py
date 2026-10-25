"""カメラを同時に open できるかを確かめる。

1 台ずつ開いて成功しても、同時に開くと失敗する構成がある。同一 USB ハブ配下で
帯域が足りないときがこれで、記録を始めてから発覚する。判定は必ず同時 open で行う。

先に開いたカメラを保持したまま次を開き、全台から実際にフレームを読めるところまで見る。
open だけ通って read が返らないカメラを弾くため。

index が「開けた」ことは、そこに意図したカメラが居ることを意味しない。ノート PC の内蔵
カメラが手首カメラの index に入り込むと、open も read も成功したまま別の映像を記録し
続ける。それを見つけるため鮮鋭度（ラプラシアン分散）を出し、--save でフレームを保存する。
ピンボケの内蔵カメラは 1 桁、ピントの合った手首カメラは数十以上になる。

Usage:
    python check_cameras.py head=0 left_wrist=1 right_wrist=2
    python check_cameras.py head=0 left_wrist=1 --save /tmp/camcheck
    CAM_BACKEND=1200 python check_cameras.py head=0 left_wrist=1   # macOS は AVFoundation

環境変数 CAM_BACKEND は cv2 のバックエンド番号（0=ANY、700=DSHOW、1200=AVFOUNDATION）。
既定は 0。Windows で index を取り違えるときは 700 を試す。

Exit code: 全台が同時に open できて読めたら 0、1 台でも落ちたら 1。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

WARMUP_FRAMES = 5  # 最初の数フレームは空で返るカメラがあるので捨てる
# ピンボケの内蔵カメラは ~1、ピントの合った手首カメラは数十以上になる
SHARPNESS_WARN = 10.0


def parse_spec(arg: str) -> tuple[str, int]:
    if "=" not in arg:
        raise SystemExit(f"'名前=index' の形で指定する: {arg}")
    name, index = arg.split("=", 1)
    return name, int(index)


def sharpness(frame) -> float:
    """ラプラシアン分散。合焦しているほど大きい。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", metavar="名前=index")
    parser.add_argument("--save", type=Path, help="各カメラの 1 フレームを保存する（目視確認用）")
    args = parser.parse_args()

    specs = [parse_spec(a) for a in args.specs]
    backend = int(os.environ.get("CAM_BACKEND", "0"))
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    opened: list[tuple[str, cv2.VideoCapture]] = []
    failures: list[str] = []
    try:
        for name, index in specs:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                # 先に開いたカメラを保持したままなので、ここでの失敗は同時 open の失敗
                failures.append(f"{name}(index={index}): open できない")
                cap.release()
                continue
            opened.append((name, cap))
            print(f"open OK: {name} (index={index})")

        for name, cap in opened:
            for _ in range(WARMUP_FRAMES):
                cap.read()
                time.sleep(0.03)
            ok, frame = cap.read()
            if not ok or frame is None:
                failures.append(f"{name}: open できたがフレームを読めない")
                continue
            h, w = frame.shape[:2]
            sharp = sharpness(frame)
            mark = "  <- ピンボケ。別のカメラが居ないか目視する" if sharp < SHARPNESS_WARN else ""
            print(f"read OK: {name} {w}x{h} 鮮鋭度={sharp:.1f}{mark}")
            if args.save:
                out = args.save / f"{name}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"         saved {out}")
    finally:
        for _, cap in opened:
            cap.release()

    for message in failures:
        print(f"  NG: {message}")
    ok = not failures and len(opened) == len(specs)
    print(f"{len(opened)}/{len(specs)} 台を同時 open  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
