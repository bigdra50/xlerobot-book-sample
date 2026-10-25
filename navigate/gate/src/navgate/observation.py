"""本書の host が流す観測（JSON）を、画像、関節、台車の速度に分ける。

host は画像を JPEG にして base64 の文字列で入れ、関節は `<名前>.pos`、台車の速度は `x.vel` などで入れる。
関節の値は、host の設定（use_degrees=False）のまま、キャリブレーションの範囲を −100〜100 にした値である。
"""

from __future__ import annotations

import base64
import binascii


def split_observation(
    obs: dict,
) -> tuple[dict[str, bytes], list[tuple[str, float]], tuple[float, float, float] | None]:
    """(カメラ名 -> JPEG, [(関節名, 値)], (vx m/s, vy m/s, wz deg/s) か None) を返す。"""
    images: dict[str, bytes] = {}
    joints: list[tuple[str, float]] = []
    for key, value in obs.items():
        if isinstance(value, str):
            if not value:
                continue  # host が JPEG にできなかったフレーム
            try:
                images[key] = base64.b64decode(value, validate=True)
            except binascii.Error:
                continue
        elif key.endswith(".pos") and isinstance(value, (int, float)):
            joints.append((key.removesuffix(".pos"), float(value)))
    joints.sort()
    keys = ("x.vel", "y.vel", "theta.vel")
    base = tuple(float(obs[k]) for k in keys) if all(k in obs for k in keys) else None
    return images, joints, base  # type: ignore[return-value]
