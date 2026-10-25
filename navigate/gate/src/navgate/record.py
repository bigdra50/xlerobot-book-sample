"""pocketsensor 端末から生ストリームを MCAP へ記録する CLI。"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pocketsensor as ps

from navgate.analysis import (
    DEFAULT_MAX_INTERVAL_S,
    DEFAULT_MAX_SPEED_MPS,
    REASON_LABEL,
    STATE_LABEL,
    PoseSample,
    TrackingSample,
    format_summary,
    summarize,
    summarize_depth,
    summarize_tracking,
)
from navgate.depth import CenterDepth
from navgate.recording import depth_center, pose_sample, tracking_sample


@dataclass(frozen=True)
class StatusSnapshot:
    """1 秒ごとの状態行に載せる値。"""

    elapsed_s: float
    poses: int
    fps: float
    position: tuple[float, float, float] | None
    tracking_state: int | None
    tracking_reason: int | None
    origin_epoch: int | None
    depth_center_m: float | None
    clock_offset_ms: float | None
    clock_rtt_ms: float | None
    depth_high_ratio: float | None = None


def format_status_line(snap: StatusSnapshot) -> str:
    """1 秒ごとの進捗行。改行は付けない（呼び出し側が ``\\r`` で上書きする）。"""
    if snap.position is None:
        pos_text = "n/a"
    else:
        x, y, z = snap.position
        pos_text = f"({x:+.2f}, {y:+.2f}, {z:+.2f})"
    if snap.tracking_state is None:
        track_text = "n/a"
    else:
        state = STATE_LABEL.get(int(snap.tracking_state), str(snap.tracking_state))
        reason = REASON_LABEL.get(int(snap.tracking_reason or 0), str(snap.tracking_reason))
        track_text = f"{state}/{reason}"
    origin_text = "n/a" if snap.origin_epoch is None else str(snap.origin_epoch)
    depth_text = "n/a" if snap.depth_center_m is None else f"{snap.depth_center_m:.2f} m"
    if snap.depth_center_m is not None and snap.depth_high_ratio is not None:
        depth_text += f" (high conf {snap.depth_high_ratio:.0%})"
    if snap.clock_offset_ms is None or snap.clock_rtt_ms is None:
        clock_text = "n/a"
    else:
        clock_text = f"offset {snap.clock_offset_ms:+.1f} ms rtt {snap.clock_rtt_ms:.1f} ms"
    return (
        f"{snap.elapsed_s:5.1f}s  poses {snap.poses:5d}  fps {snap.fps:5.1f}  "
        f"pos {pos_text}  tracking {track_text}  origin {origin_text}  "
        f"depth {depth_text}  clock {clock_text}   "
    )


def _default_out_path() -> Path:
    return Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}.mcap"


def _clock_metrics(dev: Any) -> tuple[float | None, float | None]:
    clock = dev.clock
    if not clock.ready:
        return None, None
    # wall_offset は人が読む用。offset_ns はモノトニッククロックどうしの値で桁が違う
    return clock.wall_offset_ns / 1e6, clock.rtt_ns / 1e6


def _fps_last_second(pose_times: deque[float], now: float) -> float:
    while pose_times and now - pose_times[0] > 1.0:
        pose_times.popleft()
    return float(len(pose_times))


def _open_live(source: str, depth_hz: float) -> Any:
    streams: list[Any] = [ps.Pose()]
    if depth_hz > 0:
        streams.append(ps.Depth(rate=depth_hz))
    # 深度は姿勢より遅い。ANY にしないと 1 秒の状態行が深度の間隔になる
    return ps.open(source, ps.Config(streams=tuple(streams), frame_policy=ps.FramePolicy.ANY))


def main(argv: list[str] | None = None) -> int:
    """pocketsensor 端末の姿勢を MCAP に記録する CLI（IO）。"""
    parser = argparse.ArgumentParser(description="pocketsensor 端末から姿勢を MCAP に記録する")
    parser.add_argument("--source", default="ws://iphone.local:8765", help="ws://… または usb:")
    parser.add_argument("--out", type=Path, default=None, help="出力 MCAP。省略時は runs/<日時>.mcap")
    parser.add_argument("--depth-hz", type=float, default=2.0, help="深度の購読レート。0 で購読しない")
    parser.add_argument("--duration", type=float, default=None, help="記録秒数。省略時は Ctrl+C まで")
    parser.add_argument(
        "--max-interval", type=float, default=DEFAULT_MAX_INTERVAL_S, help="途絶とみなす間隔（秒）"
    )
    parser.add_argument(
        "--max-speed", type=float, default=DEFAULT_MAX_SPEED_MPS, help="跳びとみなす速度（m/s）"
    )
    args = parser.parse_args(argv)

    out = args.out or _default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        dev = _open_live(args.source, args.depth_hz)
    except ps.ConnectionFailed as exc:
        print(str(exc), file=sys.stderr)
        return 1

    poses: list[PoseSample] = []
    tracking: list[TrackingSample] = []
    pose_times: deque[float] = deque()
    last_position: tuple[float, float, float] | None = None
    last_state: int | None = None
    last_reason: int | None = None
    last_epoch: int | None = None
    last_depth: CenterDepth | None = None
    depth_samples: list[CenterDepth | None] = []
    last_status_at = 0.0
    t_start = time.monotonic()
    try:
        with dev, dev.record(out):
            while True:
                now = time.monotonic()
                elapsed = now - t_start
                if args.duration is not None and elapsed >= args.duration:
                    break
                timeout = 0.2
                if args.duration is not None:
                    timeout = min(timeout, max(0.0, args.duration - elapsed))
                try:
                    frames = dev.wait_for_frames(timeout=timeout)
                except TimeoutError:
                    frames = None
                except ps.ConnectionLost as exc:
                    print(str(exc), file=sys.stderr)
                    break
                now = time.monotonic()
                if frames is not None:
                    pose = pose_sample(len(poses), frames)
                    if pose is not None:
                        poses.append(pose)
                        pose_times.append(now)
                        last_position = pose.position
                    status = tracking_sample(frames)
                    if status is not None:
                        tracking.append(status)
                        last_state, last_reason, last_epoch = status.state, status.reason, status.origin_epoch
                    center = depth_center(frames)
                    if center is not None:
                        last_depth = center[1]
                        depth_samples.append(last_depth)
                if now - last_status_at >= 1.0:
                    last_status_at = now
                    offset_ms, rtt_ms = _clock_metrics(dev)
                    line = format_status_line(
                        StatusSnapshot(
                            elapsed_s=now - t_start,
                            poses=len(poses),
                            fps=_fps_last_second(pose_times, now),
                            position=last_position,
                            tracking_state=last_state,
                            tracking_reason=last_reason,
                            origin_epoch=last_epoch,
                            depth_center_m=None if last_depth is None else last_depth.median_m,
                            depth_high_ratio=None if last_depth is None else last_depth.high_ratio,
                            clock_offset_ms=offset_ms,
                            clock_rtt_ms=rtt_ms,
                        )
                    )
                    print(line, end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    tracking_summary = summarize_tracking(tracking) if tracking else None
    print(
        format_summary(
            summarize(poses, max_interval_s=args.max_interval, max_speed_mps=args.max_speed),
            tracking_summary,
            summarize_depth(depth_samples) if depth_samples else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
