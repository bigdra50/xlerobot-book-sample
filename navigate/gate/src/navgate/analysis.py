"""走行テストの解析。

ARKit 姿勢の列から閉じ誤差・経路長・途絶・跳びを出す。
関数はすべて純粋で、IO は ``main`` だけが持つ（MCAP は recording.py）。
閾値はフラグ付けの既定値であり、合否は人が判断する。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

from pocketsensor.types import TrackingReason, TrackingState

from navgate.depth import CenterDepth

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# 30 fps で 6 フレーム欠けたら途絶とみなす
DEFAULT_MAX_INTERVAL_S: Final = 0.2
# 台車の最高速 0.5 m/s の 4 倍。ARKit の再局所化による跳びは桁が違うので十分に分離できる
DEFAULT_MAX_SPEED_MPS: Final = 2.0

_STATE_KEYS: Final = (
    int(TrackingState.NORMAL),
    int(TrackingState.LIMITED),
    int(TrackingState.NOT_AVAILABLE),
)

STATE_LABEL: Final = {
    int(TrackingState.NORMAL): "normal",
    int(TrackingState.LIMITED): "limited",
    int(TrackingState.NOT_AVAILABLE): "unavailable",
}

REASON_LABEL: Final = {
    int(TrackingReason.NONE): "none",
    int(TrackingReason.INITIALIZING): "initializing",
    int(TrackingReason.EXCESSIVE_MOTION): "excessive motion",
    int(TrackingReason.INSUFFICIENT_FEATURES): "insufficient features",
    int(TrackingReason.RELOCALIZING): "relocalizing",
}


@dataclass(frozen=True)
class PoseSample:
    """1 フレーム分の姿勢。

    Attributes:
        seq: 連番。
        t: 解析に使う時刻（秒）。odom の ``header.stamp``
            （端末の計測時刻をアプリがシステム時刻へ換算した値。到着時刻ではない。到着時刻は
            WiFi の揺れを途絶や跳びとして数えてしまう）。
        position: ``<name>_odom`` の座標（REP-103: x 前、y 左、z 上、m）。
        orientation: クォータニオン (qx, qy, qz, qw)。
    """

    seq: int
    t: float
    position: Vec3
    orientation: Quat


@dataclass(frozen=True)
class TrackingSample:
    """1 フレーム分のトラッキング状態。時刻は姿勢と同じく ``header.stamp``（秒）。"""

    t: float
    state: int
    reason: int
    origin_epoch: int


@dataclass(frozen=True)
class Gap:
    """フレーム間隔が閾値を超えた区間。"""

    seq_before: int
    seq_after: int
    dt: float


@dataclass(frozen=True)
class Jump:
    """速度が閾値を超えたフレーム（トラッキングの跳び・再局所化の候補）。"""

    seq: int
    speed: float


@dataclass(frozen=True)
class RunSummary:
    """1 回の走行の要約。"""

    frames: int
    duration_s: float
    mean_fps: float
    path_length_m: float
    closure_error_m: float
    max_distance_from_start_m: float
    gaps: tuple[Gap, ...]
    jumps: tuple[Jump, ...]


@dataclass(frozen=True)
class TrackingSummary:
    """トラッキング状態の時間割合と原点リセットの回数。"""

    duration_s: float
    share_normal: float
    share_limited: float
    share_unavailable: float
    limited_reasons: dict[int, int]
    origin_epoch_changes: int


def path_length(positions: Sequence[Vec3]) -> float:
    """隣接点間距離の総和。"""
    return sum(math.dist(a, b) for a, b in pairwise(positions))


def closure_error(positions: Sequence[Vec3]) -> float:
    """最初と最後の位置の距離。始点へ戻る走行ではドリフトの指標になる。"""
    if len(positions) < 2:
        return 0.0
    return math.dist(positions[0], positions[-1])


def max_distance_from_start(positions: Sequence[Vec3]) -> float:
    """始点から最も離れた点までの距離。走行の広がりの確認用。"""
    if not positions:
        return 0.0
    start = positions[0]
    return max(math.dist(start, p) for p in positions)


def find_gaps(
    samples: Sequence[PoseSample], max_interval_s: float = DEFAULT_MAX_INTERVAL_S
) -> tuple[Gap, ...]:
    """フレーム間隔が ``max_interval_s`` を超えた区間を列挙する。"""
    return tuple(
        Gap(seq_before=a.seq, seq_after=b.seq, dt=b.t - a.t)
        for a, b in pairwise(samples)
        if b.t - a.t > max_interval_s
    )


def find_jumps(
    samples: Sequence[PoseSample], max_speed_mps: float = DEFAULT_MAX_SPEED_MPS
) -> tuple[Jump, ...]:
    """隣接フレーム間の速度が ``max_speed_mps`` を超えたフレームを列挙する。

    同時刻のフレーム（dt <= 0）は速度が定義できないので飛ばす。
    """
    jumps: list[Jump] = []
    for a, b in pairwise(samples):
        dt = b.t - a.t
        if dt <= 0:
            continue
        speed = math.dist(a.position, b.position) / dt
        if speed > max_speed_mps:
            jumps.append(Jump(seq=b.seq, speed=speed))
    return tuple(jumps)


def summarize(
    samples: Sequence[PoseSample],
    *,
    max_interval_s: float = DEFAULT_MAX_INTERVAL_S,
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
) -> RunSummary:
    """走行 1 回分の要約を作る。"""
    positions = [s.position for s in samples]
    duration = samples[-1].t - samples[0].t if len(samples) >= 2 else 0.0
    mean_fps = (len(samples) - 1) / duration if duration > 0 else 0.0
    return RunSummary(
        frames=len(samples),
        duration_s=duration,
        mean_fps=mean_fps,
        path_length_m=path_length(positions),
        closure_error_m=closure_error(positions),
        max_distance_from_start_m=max_distance_from_start(positions),
        gaps=find_gaps(samples, max_interval_s),
        jumps=find_jumps(samples, max_speed_mps),
    )


def summarize_tracking(samples: Sequence[TrackingSample]) -> TrackingSummary:
    """トラッキング状態を、次サンプルまでの間隔で重み付けして集計する。"""
    ordered = sorted(samples, key=lambda s: s.t)
    reasons: Counter[int] = Counter()
    for sample in ordered:
        if int(sample.state) == int(TrackingState.LIMITED):
            reasons[int(sample.reason)] += 1
    origin_changes = sum(1 for a, b in pairwise(ordered) if int(b.origin_epoch) != int(a.origin_epoch))
    weights = {
        int(TrackingState.NORMAL): 0.0,
        int(TrackingState.LIMITED): 0.0,
        int(TrackingState.NOT_AVAILABLE): 0.0,
    }
    total = 0.0
    for a, b in pairwise(ordered):
        dt = b.t - a.t
        if dt <= 0:
            continue
        state = int(a.state)
        if state not in weights:
            weights[state] = 0.0
        weights[state] += dt
        total += dt
    duration = ordered[-1].t - ordered[0].t if len(ordered) >= 2 else 0.0
    if total <= 0.0 and ordered:
        # 間隔が無い（1 点、または同時刻）ときは、その状態が走行全体を代表する
        weights = {
            int(TrackingState.NORMAL): 0.0,
            int(TrackingState.LIMITED): 0.0,
            int(TrackingState.NOT_AVAILABLE): 0.0,
        }
        weights[int(ordered[0].state)] = 1.0
        total = 1.0
    share = {key: (weights.get(key, 0.0) / total if total > 0 else 0.0) for key in _STATE_KEYS}
    return TrackingSummary(
        duration_s=duration,
        share_normal=share[int(TrackingState.NORMAL)],
        share_limited=share[int(TrackingState.LIMITED)],
        share_unavailable=share[int(TrackingState.NOT_AVAILABLE)],
        limited_reasons=dict(reasons),
        origin_epoch_changes=origin_changes,
    )


@dataclass(frozen=True)
class DepthSummary:
    """深度の生存確認。

    Attributes:
        frames: 深度のフレーム数。
        with_distance: 中央パッチに有効な距離があったフレーム数。
        median_m: 中央パッチの距離の中央値（m）。
        high_ratio: 高信頼の画素の割合の平均。confidence の無い記録では None。
    """

    frames: int
    with_distance: int
    median_m: float | None
    high_ratio: float | None


def summarize_depth(samples: Sequence[CenterDepth | None]) -> DepthSummary:
    """フレームごとの中央パッチの結果をまとめる（純粋）。"""
    found = [sample for sample in samples if sample is not None]
    ratios = [sample.high_ratio for sample in found if sample.high_ratio is not None]
    return DepthSummary(
        frames=len(samples),
        with_distance=len(found),
        median_m=statistics.median(sample.median_m for sample in found) if found else None,
        high_ratio=statistics.fmean(ratios) if ratios else None,
    )


def _format_depth(depth: DepthSummary) -> str:
    if depth.median_m is None:
        return f"depth: {depth.frames} frames, no valid distance at the center"
    text = f"depth: {depth.frames} frames, center {depth.median_m:.2f} m"
    if depth.high_ratio is not None:
        text += f", high confidence {depth.high_ratio:.0%}"
    return text


def format_summary(
    summary: RunSummary,
    tracking: TrackingSummary | None = None,
    depth: DepthSummary | None = None,
) -> str:
    """人が読む要約テキスト。"""
    ratio = summary.closure_error_m / summary.path_length_m if summary.path_length_m > 0 else 0.0
    lines = [
        f"frames: {summary.frames}  duration: {summary.duration_s:.1f} s  mean fps: {summary.mean_fps:.1f}",
        f"path length: {summary.path_length_m:.2f} m  max distance from start: {summary.max_distance_from_start_m:.2f} m",
        f"closure error: {summary.closure_error_m:.3f} m ({ratio:.1%} of path)",
        f"gaps: {len(summary.gaps)}",
        *(f"  seq {g.seq_before} -> {g.seq_after}: {g.dt:.2f} s" for g in summary.gaps),
        f"jumps: {len(summary.jumps)}",
        *(f"  seq {j.seq}: {j.speed:.1f} m/s" for j in summary.jumps),
    ]
    if tracking is not None:
        lines.append(
            "tracking: "
            f"normal {tracking.share_normal:.1%}  "
            f"limited {tracking.share_limited:.1%}  "
            f"unavailable {tracking.share_unavailable:.1%}"
        )
        if tracking.limited_reasons:
            parts = [
                f"{REASON_LABEL.get(reason, str(reason))} {count}"
                for reason, count in sorted(tracking.limited_reasons.items())
            ]
            lines.append("limited reasons: " + ", ".join(parts))
        lines.append(f"origin_epoch changes: {tracking.origin_epoch_changes}")
        if tracking.origin_epoch_changes:
            # 原点が途中で作り直されると始点と終点が別の world になり、閉じ誤差は意味を持たない
            lines.append("warning: origin_epoch changed during the run; closure error is meaningless")
    if depth is not None:
        lines.append(_format_depth(depth))
    return "\n".join(lines)


def _summarize_path(
    path: Path,
    *,
    max_interval_s: float,
    max_speed_mps: float,
) -> str:
    suffix = path.suffix.lower()
    tracking: TrackingSummary | None = None
    depth: DepthSummary | None = None
    if suffix == ".mcap":
        from navgate.recording import read_mcap

        recording = read_mcap(path)
        samples = recording.poses
        if recording.tracking:
            tracking = summarize_tracking(recording.tracking)
        if recording.depth_center:
            depth = summarize_depth([center for _t, center in recording.depth_center])
    else:
        raise ValueError(f"unsupported recording type: {path.suffix}")
    return format_summary(
        summarize(samples, max_interval_s=max_interval_s, max_speed_mps=max_speed_mps),
        tracking,
        depth,
    )


def main(argv: list[str] | None = None) -> int:
    """pocketsensor の MCAP を要約する CLI。"""
    parser = argparse.ArgumentParser(description="走行の記録（MCAP）を要約する")
    parser.add_argument("path", type=Path, help="gate-record の MCAP")
    parser.add_argument(
        "--max-interval", type=float, default=DEFAULT_MAX_INTERVAL_S, help="途絶とみなす間隔（秒）"
    )
    parser.add_argument(
        "--max-speed", type=float, default=DEFAULT_MAX_SPEED_MPS, help="跳びとみなす速度（m/s）"
    )
    args = parser.parse_args(argv)
    try:
        text = _summarize_path(args.path, max_interval_s=args.max_interval, max_speed_mps=args.max_speed)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
