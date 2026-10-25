"""天板の手前の縁を、深度から作った 3D の点で見つける（純粋関数）。

座標は base_link（x が前、y が左、z が上。原点は床の上の、その場で回るときの回転の中心）。
iPad のレンズ（高さ 0.53 m）は天板より低いので、深度に映るのは縁の面と天板の裏側になる。
天板の高さの帯にある点を y で区切り、区切りごとに一番手前の x を取って、手前の縁の線を当てはめる。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Edge:
    """手前の縁の線 x = offset + slope * y と、見えている範囲。"""

    offset: float  # y = 0（台車の前方の軸）での縁の x [m]
    slope: float
    y_right: float  # 縁の点が見えている範囲 [m]。視野の端で切れていることがある
    y_left: float
    height: float  # 縁の点の高さの中央値 [m]
    bins: int  # 当てはめに使った y の区切りの数
    rms: float  # 当てはめの残差の RMS [m]
    # 縁の真下（帯より低い高さ）にも点がある区切りの割合。壁なら 1 に近く、張り出した天板なら 0 に近い
    support_below: float
    # 縁の端の先が視野の端で切れている（本当の端かどうか分からない）。camera を渡さなければ常に False
    clipped_right: bool = False
    clipped_left: bool = False

    @property
    def yaw(self) -> float:
        """縁に正対するまでに回る角 [rad]。左回りが正。"""
        return -math.atan(self.slope)

    @property
    def distance(self) -> float:
        """base_link の原点から縁の線までの距離 [m]。正対したときの、原点から縁までの x に当たる。"""
        return self.offset / math.hypot(1.0, self.slope)


@dataclass(frozen=True)
class Camera:
    """深度のカメラ。K は (fx, fy, cx, cy)、T_base_opt は base_link から見た光学 frame（z 前、x 右、y 下）の 4×4。"""

    K: Sequence[float]
    width: int
    height: int
    T_base_opt: ArrayLike

    def contains(self, points: ArrayLike, margin_px: float = 2.0) -> NDArray[np.bool_]:
        """base_link の点が、画像の内側（端から margin_px 以上）に写るかどうか。"""
        fx, fy, cx, cy = self.K
        T = np.asarray(self.T_base_opt, dtype=np.float64)
        opt = (np.asarray(points, dtype=np.float64) - T[:3, 3]) @ T[:3, :3]
        z = opt[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * opt[:, 0] / z + cx
            v = fy * opt[:, 1] / z + cy
        return (
            (z > 0)
            & (u >= margin_px)
            & (u <= self.width - 1 - margin_px)
            & (v >= margin_px)
            & (v <= self.height - 1 - margin_px)
        )


def points_from_depth(
    depth_m: NDArray[np.floating], K: Sequence[float], T_base_opt: ArrayLike
) -> NDArray[np.float64]:
    """深度の画像 [m] を base_link の点（N×3）にする。0 と NaN の画素は捨てる。

    K は (fx, fy, cx, cy)、T_base_opt は base_link から見た光学 frame（z 前、x 右、y 下）の 4×4。
    """
    fx, fy, cx, cy = K
    z = np.asarray(depth_m, dtype=np.float64)
    v, u = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    ok = np.isfinite(z) & (z > 0.0)
    opt = np.stack([(u[ok] - cx) * z[ok] / fx, (v[ok] - cy) * z[ok] / fy, z[ok]], axis=-1)
    T = np.asarray(T_base_opt, dtype=np.float64)
    return opt @ T[:3, :3].T + T[:3, 3]


def min_visible_distance(
    camera: Camera, height: float, *, margin_px: float = 4.0, clip_check: float = 0.03
) -> float:
    """高さ height [m] の縁を find_edge が見られる、レンズから縁までの一番近い水平距離 [m]。

    縁がレンズより高いと、近づくほど縁は視野の上へ外れる。画像の上から margin_px の行が見上げる角で決まり、
    find_edge が縁を clip_check だけ手前へずらして視野の内側かを確かめる分を足す。
    """
    _, fy, _, cy = camera.K
    T = np.asarray(camera.T_base_opt, dtype=np.float64)
    ray = T[:3, :3] @ np.array([0.0, (margin_px - cy) / fy, 1.0])
    elevation = math.atan2(ray[2], math.hypot(ray[0], ray[1]))
    rise = height - T[2, 3]
    if rise <= 0.0:
        return clip_check
    if elevation <= 0.0:
        return math.inf
    return rise / math.tan(elevation) + clip_check


def _fit(y: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[float, float]:
    if len(y) < 2:
        return float(x[0]), 0.0
    A = np.stack([np.ones(len(y)), y], axis=-1)
    (a, s), *_ = np.linalg.lstsq(A, x, rcond=None)
    return float(a), float(s)


def find_edge(
    points: ArrayLike,
    *,
    band: tuple[float, float] = (0.55, 0.95),
    x_range: tuple[float, float] = (0.15, 2.5),
    y_limit: float = 0.8,
    bin_width: float = 0.02,
    min_points: int = 4,
    tol: float = 0.025,
    min_bins: int = 6,
    below_from: float = 0.10,
    camera: Camera | None = None,
    clip_check: float = 0.03,
) -> Edge | None:
    """帯（天板の高さ）の中で一番手前にあり、横に min_bins 区切り以上続く線を返す。無ければ None。

    帯より低い点は縁に数えない。天板の縁は床から離れているので、真下に点があるかどうかで壁と見分けられる。
    camera を渡すと、視野の端で切れた区切りを捨てる。縁の見本を clip_check [m] だけ手前へずらした点が画像の外へ出るなら、
    その区切りで見えている一番手前は縁ではなく視野の端であり、縁は見えていない。
    """
    p = np.asarray(points, dtype=np.float64)
    sel = (
        (p[:, 2] >= band[0])
        & (p[:, 2] <= band[1])
        & (p[:, 0] >= x_range[0])
        & (p[:, 0] <= x_range[1])
        & (np.abs(p[:, 1]) <= y_limit)
    )
    q = p[sel]
    if len(q) == 0:
        return None
    n_bins = round(2 * y_limit / bin_width)
    idx = np.clip(np.floor((q[:, 1] + y_limit) / bin_width).astype(int), 0, n_bins - 1)
    # 区切りごとの縁の見本。手前の縁の点の、平均の (y, x) にする。
    # 区切りの中心の y を使うと、縁が傾いているときに、区切りの中の点の並び方で傾きがずれる
    near = np.full(n_bins, np.nan)
    ys = np.full(n_bins, np.nan)
    zs = np.full(n_bins, np.nan)
    members: dict[int, NDArray[np.float64]] = {}
    for b in np.unique(idx):
        pts = q[idx == b]
        if len(pts) < min_points:
            continue
        # 手前から数えて min_points 番目（多い区切りでは 1 %）の点の前後 5 mm に限り、飛び離れた数点で縁を手前へ引かない。
        # 割合だけで決めると、天板の裏側の点が多い区切りで、縁より奥の点を拾う
        k = max(min_points - 1, int(0.01 * len(pts)))
        ref = np.partition(pts[:, 0], k)[k]
        front = pts[np.abs(pts[:, 0] - ref) <= 0.005]
        near[b], ys[b], zs[b] = front[:, 0].mean(), front[:, 1].mean(), front[:, 2].mean()
        members[int(b)] = front
    clipped = np.zeros(n_bins, dtype=bool)
    if camera is not None:
        has = ~np.isnan(near)
        probe = np.stack([near - clip_check, ys, zs], axis=-1)
        clipped[has] = ~camera.contains(probe[has])
        near[clipped] = np.nan

    def grow(seed: int) -> list[int]:
        # 始点から左右へ、線の予測に合う区切りを足していく。空の区切りは 1 つまで飛び越す
        group = [seed]
        for step in (1, -1):
            b, misses = seed, 0
            while 0 <= b + step < n_bins:
                b += step
                if np.isnan(near[b]):
                    misses += 1
                    if misses > 1:
                        break
                    continue
                g = np.array(sorted(group))
                a, s = _fit(ys[g], near[g])
                if abs(near[b] - (a + s * ys[b])) > tol:
                    break
                group.append(b)
                misses = 0
        return sorted(group)

    # 始点は、手前にあって隣の区切りとも合う区切り。孤立した 1 区切りでは始めない。
    # 手前から順に試し、min_bins に届かない細い物（縁より手前に立つ柱など）は飛ばして次を試す
    tried: set[int] = set()
    group: list[int] = []
    for b in np.argsort(np.where(np.isnan(near), np.inf, near)):
        b = int(b)
        if np.isnan(near[b]):
            break
        if b in tried:
            continue
        neighbours = [near[j] for j in (b - 1, b + 1) if 0 <= j < n_bins and not np.isnan(near[j])]
        if not any(abs(v - near[b]) <= tol for v in neighbours):
            continue
        candidate = grow(b)
        tried.update(candidate)
        if len(candidate) >= min_bins:
            group = candidate
            break
    if not group:
        return None
    g = np.array(group)
    a, s = _fit(ys[g], near[g])
    res = near[g] - (a + s * ys[g])
    keep = g[np.abs(res) <= 0.015]
    if min_bins <= len(keep) < len(g):
        g = keep
        a, s = _fit(ys[g], near[g])
        res = near[g] - (a + s * ys[g])

    def cut_beyond(end: int, step: int) -> bool:
        # 端の先の区切り（空を 1 つまで飛ばす）が、視野の端で捨てた区切りなら True
        for b in (end + step, end + 2 * step):
            if not 0 <= b < n_bins:
                return False
            if clipped[b]:
                return True
            if not np.isnan(near[b]):
                return False
        return False

    front = np.concatenate([members[int(b)] for b in g])
    low = p[(p[:, 2] >= below_from) & (p[:, 2] < band[0])]
    low_near = low[np.abs(low[:, 0] - (a + s * low[:, 1])) <= 0.03]
    low_idx = np.floor((low_near[:, 1] + y_limit) / bin_width).astype(int)
    supported = sum(int(np.count_nonzero(low_idx == b) >= min_points) for b in g)
    return Edge(
        offset=a,
        slope=s,
        y_right=float(front[:, 1].min()),
        y_left=float(front[:, 1].max()),
        height=float(np.median(front[:, 2])),
        bins=len(g),
        rms=float(np.sqrt(np.mean(res**2))),
        support_below=supported / len(g),
        clipped_right=cut_beyond(int(g[0]), -1),
        clipped_left=cut_beyond(int(g[-1]), 1),
    )


def wall_ahead(
    points: ArrayLike,
    *,
    lens_y: float,
    beyond: float,
    band: tuple[float, float] = (0.20, 0.50),
    half_width: float = 0.10,
    bin_width: float = 0.02,
    min_points: int = 10,
) -> float | None:
    """レンズの正面の、縁より奥で低い高さに見える面（机の下の壁など）までの x [m]。点が足りなければ None。

    天板の縁の深度は、黒く薄い縁ほど距離によって 1〜2 cm ずれる。広い面の深度はずれが小さいので、
    縁が視野の上へ外れたあとに、台車がどれだけ進んだかをこの面で測る。
    手前の支柱は画素が多く写るので、点の中央値では支柱に寄る。y の区切りごとの中央値の、さらに中央値を取る。
    """
    p = np.asarray(points, dtype=np.float64)
    sel = (
        (p[:, 2] >= band[0])
        & (p[:, 2] <= band[1])
        & (np.abs(p[:, 1] - lens_y) <= half_width)
        & (p[:, 0] > beyond)
    )
    q = p[sel]
    idx = np.floor((q[:, 1] - lens_y + half_width) / bin_width).astype(int)
    per_bin = [np.median(q[idx == b, 0]) for b in np.unique(idx) if np.count_nonzero(idx == b) >= min_points]
    if len(per_bin) < max(3, round(half_width / bin_width)):
        return None
    return float(np.median(per_bin))


def nearest_in_corridor(
    points: ArrayLike, *, half_width: float, band: tuple[float, float] = (0.05, 1.3), k: int = 10
) -> float | None:
    """台車の通り道（|y| <= half_width）で、床より高い点のうち一番手前の x [m]。k 番目の点にして、飛び離れた点を拾わない。"""
    p = np.asarray(points, dtype=np.float64)
    x = p[(np.abs(p[:, 1]) <= half_width) & (p[:, 2] >= band[0]) & (p[:, 2] <= band[1]), 0]
    if len(x) <= k:
        return None
    return float(np.partition(x, k)[k])


def approach_distance(edge: Edge, *, front_x: float, gap: float) -> float:
    """正対したあと、台車の前の縁（base_link の x = front_x）と天板の縁のあいだを gap にするまでに進む距離 [m]。"""
    return edge.distance - front_x - gap


def docked_gap(*, distance_after: float, backoff: float, front_x: float) -> float:
    """寄せたあと backoff [m] 下がって測った縁の距離から、寄せていたときの隙間 [m] を求める。"""
    return distance_after - backoff - front_x
