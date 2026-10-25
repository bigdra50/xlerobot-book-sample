"""名前を付けた場所と、その場所へ最後の数十 cm を寄せる手順。

場所は、最終の位置（pose）と待機位置（entry）を持つ。どちらも map の座標で (x, y, yaw_deg)。
待機位置までは Nav2 で行き、待機位置から最終の位置までは、次の 1 手を毎回 next_step で決めて動く。
Nav2 の制御は前進と旋回だけなので、机の横のような、その場で回れない場所へは入れない。
そこで待機位置（回れる場所）で向きをそろえ、横移動で入る。

1 手の決め方（場所の向きを前とする座標で考える）:
1. 向きがずれていれば回る
2. 横にずれていれば、まず standoff [m] 手前の線まで下がり、そこで横へ動く（机の縁から離れて横移動するため）
3. 横が合えば、前へ進んで場所に着く
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from navgate.marker_align import MarkerRef

TURN_TOL_DEG = 1.5
LAT_TOL_M = 0.01
LONG_TOL_M = 0.005
MAX_SLIDE_M = 0.25
# 待機位置からこれより離れていたら、待機位置まで Nav2 で行っていないとみなす
MAX_LONG_M = 1.0
MAX_LAT_M = 1.5


@dataclass(frozen=True)
class Place:
    name: str
    pose: tuple[float, float, float]
    entry: tuple[float, float, float]
    standoff: float
    # 着いたあとに合わせ込む、壁のマーカーの基準。無ければ地図の上の位置だけで止まる
    marker: MarkerRef | None = None


@dataclass(frozen=True)
class Step:
    kind: str  # "turn"（deg、左が正）、"fwd"（m、前が正）、"left"（m、左が正）
    amount: float


def load_places(path: Path) -> dict[str, Place]:
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    return {
        name: Place(
            name=name,
            pose=tuple(v["pose"]),
            entry=tuple(v["entry"]),
            standoff=float(v["standoff"]),
            marker=_marker_from(v.get("marker")),
        )
        for name, v in raw.items()
    }


def _marker_from(v: dict | None) -> MarkerRef | None:
    if not v:
        return None
    return MarkerRef(
        marker_id=int(v["marker_id"]), marker_xy=tuple(v["marker_xy"]), wall_yaw=float(v["wall_yaw"])
    )


def save_place(path: Path, place: Place) -> None:
    places = load_places(path)
    places[place.name] = place
    out = {
        n: {k: v for k, v in asdict(p).items() if k != "name" and not (k == "marker" and v is None)}
        for n, p in sorted(places.items())
    }
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def relative(current: tuple[float, float, float], place: Place) -> tuple[float, float, float]:
    """今の位置を、場所の座標（前 long、左 lat）で表す。dyaw は今の向きから場所の向きを引いた度。"""
    x, y, yaw = current
    px, py, pyaw = place.pose
    h = math.radians(pyaw)
    dx, dy = x - px, y - py
    long = dx * math.cos(h) + dy * math.sin(h)
    lat = -dx * math.sin(h) + dy * math.cos(h)
    return long, lat, _wrap_deg(yaw - pyaw)


def next_step(current: tuple[float, float, float], place: Place) -> Step | None:
    """次の 1 手。着いていれば None。遠すぎれば ValueError（待機位置まで Nav2 で行く）。"""
    long, lat, dyaw = relative(current, place)
    if abs(long) > MAX_LONG_M or abs(lat) > MAX_LAT_M:
        raise ValueError(f"場所から遠すぎる（前後 {long:+.2f} m、左右 {lat:+.2f} m）。先に待機位置へ行く")
    if abs(dyaw) > TURN_TOL_DEG:
        return Step("turn", -dyaw)
    if abs(lat) > LAT_TOL_M:
        if long > -place.standoff + 0.02:
            return Step("fwd", -(place.standoff + long))
        return Step("left", max(-MAX_SLIDE_M, min(MAX_SLIDE_M, -lat)))
    if abs(long) > LONG_TOL_M:
        return Step("fwd", -long)
    return None


def exit_of(place: Place) -> Place:
    """場所から待機位置へ出るための、待機位置を最終の位置とする場所。入った手順を逆にたどる。"""
    return Place(name=f"{place.name}:exit", pose=place.entry, entry=place.entry, standoff=place.standoff)


def near(current: tuple[float, float, float], place: Place, radius: float) -> bool:
    return math.hypot(current[0] - place.pose[0], current[1] - place.pose[1]) <= radius
