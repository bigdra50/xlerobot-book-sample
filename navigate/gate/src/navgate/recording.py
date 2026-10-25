"""pocketsensor の MCAP 記録を姿勢・トラッキング・深度中心へ落とす（IO）。

再生は SDK の ``wait_for_frames`` を ``realtime=False`` で回し、EOFError で終わる。
トピック接頭辞は端末名なので、ここでは名前を直書きしない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pocketsensor as ps

from navgate.analysis import PoseSample, TrackingSample
from navgate.depth import CenterDepth, center_patch_depth


@dataclass(frozen=True)
class Recording:
    """1 ファイル分の記録。"""

    poses: list[PoseSample]
    tracking: list[TrackingSample]
    depth_center: list[tuple[float, CenterDepth | None]]


def _open_recording(path: Path) -> ps.Device:
    source = str(path)
    # 姿勢と深度のレートが違うので、欠けた組も返す。REQUIRE_ALL だと深度の無い姿勢が落ちる
    policy = ps.FramePolicy.ANY
    try:
        return ps.open(
            source,
            ps.Config(streams=(ps.Pose(), ps.Depth()), frame_policy=policy),
            realtime=False,
        )
    except ps.Unsupported:
        return ps.open(
            source,
            ps.Config(streams=(ps.Pose(),), frame_policy=policy),
            realtime=False,
        )


def _stamp_s(frames: ps.FrameSet) -> float:
    # t_device_ns は再生時も odom の header.stamp（アプリがシステム時刻へ換算した計測時刻）
    return frames.t_device_ns / 1e9


def pose_sample(seq: int, frames: ps.FrameSet) -> PoseSample | None:
    pose = frames.pose
    if pose is None:
        return None
    position = pose.position
    orientation = pose.orientation_xyzw
    return PoseSample(
        seq=seq,
        t=_stamp_s(frames),
        position=(float(position[0]), float(position[1]), float(position[2])),
        orientation=(
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        ),
    )


def tracking_sample(frames: ps.FrameSet) -> TrackingSample | None:
    status = frames.tracking
    if status is None:
        return None
    return TrackingSample(
        t=_stamp_s(frames),
        state=int(status.state),
        reason=int(status.reason),
        origin_epoch=int(status.origin_epoch),
    )


def depth_center(frames: ps.FrameSet) -> tuple[float, CenterDepth | None] | None:
    depth = frames.depth
    if depth is None:
        return None
    confidence = None if frames.confidence is None else frames.confidence.levels
    return _stamp_s(frames), center_patch_depth(depth.meters, confidence)


def read_mcap(path: Path) -> Recording:
    """MCAP を最後まで読み、姿勢・トラッキング・深度中心を返す。"""
    poses: list[PoseSample] = []
    tracking: list[TrackingSample] = []
    centers: list[tuple[float, CenterDepth | None]] = []
    seq = 0
    with _open_recording(path) as dev:
        while True:
            try:
                frames = dev.wait_for_frames()
            except EOFError:
                break
            pose = pose_sample(seq, frames)
            if pose is not None:
                poses.append(pose)
                seq += 1
            sample = tracking_sample(frames)
            if sample is not None:
                tracking.append(sample)
            center = depth_center(frames)
            if center is not None:
                centers.append(center)
    return Recording(poses=poses, tracking=tracking, depth_center=centers)
