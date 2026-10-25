"""走行テストの解析（純粋関数）の仕様。

閉じ誤差・経路長・途絶・跳びを ARKit 姿勢の列から出す。
閾値はフラグ付けの既定値であり合否の判定はしない（判定は人が行う）。
"""

from __future__ import annotations

import math

import pytest
from pocketsensor.types import TrackingReason, TrackingState

from navgate.analysis import (
    DepthSummary,
    PoseSample,
    TrackingSample,
    closure_error,
    find_gaps,
    find_jumps,
    format_summary,
    max_distance_from_start,
    path_length,
    summarize,
    summarize_depth,
    summarize_tracking,
)
from navgate.depth import CenterDepth

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _sample(seq: int, t: float, x: float, y: float = 0.0, z: float = 0.0) -> PoseSample:
    return PoseSample(seq=seq, t=t, position=(x, y, z), orientation=IDENTITY)


def _square_run(step: float = 1 / 30) -> list[PoseSample]:
    """2 m 四方を 1 周して原点へ戻る 5 点の走行。"""
    corners = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    return [_sample(i, i * step, x, 0.0, z) for i, (x, z) in enumerate(corners)]


def test_path_length_of_square_is_perimeter() -> None:
    positions = [s.position for s in _square_run()]
    assert path_length(positions) == pytest.approx(8.0)


def test_path_length_of_single_point_is_zero() -> None:
    assert path_length([(1.0, 2.0, 3.0)]) == 0.0


def test_closure_error_is_zero_when_returned_to_start() -> None:
    positions = [s.position for s in _square_run()]
    assert closure_error(positions) == pytest.approx(0.0)


def test_closure_error_is_distance_between_last_and_first() -> None:
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.1, 0.0, 0.0)]
    assert closure_error(positions) == pytest.approx(0.1)


def test_max_distance_from_start_uses_farthest_point() -> None:
    positions = [s.position for s in _square_run()]
    assert max_distance_from_start(positions) == pytest.approx(math.sqrt(8.0))


def test_find_gaps_flags_intervals_above_threshold() -> None:
    samples = [_sample(0, 0.0, 0.0), _sample(1, 0.033, 0.0), _sample(2, 0.5, 0.0), _sample(3, 0.533, 0.0)]
    gaps = find_gaps(samples, max_interval_s=0.2)
    assert len(gaps) == 1
    assert (gaps[0].seq_before, gaps[0].seq_after) == (1, 2)
    assert gaps[0].dt == pytest.approx(0.467)


def test_find_gaps_returns_empty_for_steady_stream() -> None:
    assert find_gaps(_square_run(), max_interval_s=0.2) == ()


def test_find_jumps_flags_speed_above_threshold() -> None:
    samples = [_sample(0, 0.0, 0.0), _sample(1, 1 / 30, 0.01), _sample(2, 2 / 30, 1.01)]
    jumps = find_jumps(samples, max_speed_mps=2.0)
    assert [j.seq for j in jumps] == [2]
    assert jumps[0].speed == pytest.approx(30.0)


def test_find_jumps_ignores_zero_dt() -> None:
    samples = [_sample(0, 0.0, 0.0), _sample(1, 0.0, 5.0)]
    assert find_jumps(samples, max_speed_mps=2.0) == ()


def test_summarize_square_run() -> None:
    summary = summarize(_square_run(), max_interval_s=0.2, max_speed_mps=1000.0)
    assert summary.frames == 5
    assert summary.duration_s == pytest.approx(4 / 30)
    assert summary.mean_fps == pytest.approx(30.0)
    assert summary.path_length_m == pytest.approx(8.0)
    assert summary.closure_error_m == pytest.approx(0.0)
    assert summary.max_distance_from_start_m == pytest.approx(math.sqrt(8.0))
    assert summary.gaps == ()
    assert summary.jumps == ()


def test_summarize_empty_run_has_zero_frames() -> None:
    summary = summarize([])
    assert summary.frames == 0
    assert summary.duration_s == 0.0
    assert summary.mean_fps == 0.0


def test_format_summary_reports_key_numbers() -> None:
    text = format_summary(summarize(_square_run(), max_speed_mps=1000.0))
    assert "8.00" in text  # 経路長
    assert "frames" in text
    assert "closure" in text


def test_summarize_tracking_weights_each_sample_by_interval_to_next() -> None:
    samples = [
        TrackingSample(0.0, int(TrackingState.NORMAL), int(TrackingReason.NONE), 0),
        TrackingSample(1.0, int(TrackingState.LIMITED), int(TrackingReason.EXCESSIVE_MOTION), 0),
        TrackingSample(3.0, int(TrackingState.NOT_AVAILABLE), int(TrackingReason.NONE), 0),
    ]
    summary = summarize_tracking(samples)
    assert summary.share_normal == pytest.approx(1 / 3)
    assert summary.share_limited == pytest.approx(2 / 3)
    assert summary.share_unavailable == pytest.approx(0.0)
    assert summary.limited_reasons[int(TrackingReason.EXCESSIVE_MOTION)] == 1
    assert summary.origin_epoch_changes == 0


def test_summarize_tracking_counts_origin_epoch_changes() -> None:
    samples = [
        TrackingSample(0.0, int(TrackingState.NORMAL), 0, 0),
        TrackingSample(0.1, int(TrackingState.NORMAL), 0, 0),
        TrackingSample(0.2, int(TrackingState.LIMITED), int(TrackingReason.RELOCALIZING), 1),
        TrackingSample(0.3, int(TrackingState.NORMAL), 0, 2),
    ]
    summary = summarize_tracking(samples)
    assert summary.origin_epoch_changes == 2
    assert summary.limited_reasons[int(TrackingReason.RELOCALIZING)] == 1


def test_summarize_tracking_empty_and_single_sample() -> None:
    empty = summarize_tracking([])
    assert empty.share_normal == 0.0
    assert empty.origin_epoch_changes == 0
    one = summarize_tracking([TrackingSample(1.0, int(TrackingState.LIMITED), 3, 7)])
    assert one.share_limited == pytest.approx(1.0)
    assert one.share_normal == 0.0
    assert one.limited_reasons[3] == 1
    assert one.origin_epoch_changes == 0


def test_format_summary_includes_tracking_and_warns_on_origin_change() -> None:
    poses = _square_run()
    tracking = summarize_tracking(
        [
            TrackingSample(0.0, int(TrackingState.NORMAL), 0, 0),
            TrackingSample(0.1, int(TrackingState.LIMITED), int(TrackingReason.INITIALIZING), 1),
        ]
    )
    text = format_summary(summarize(poses, max_speed_mps=1000.0), tracking)
    assert "tracking" in text
    assert "origin_epoch" in text
    assert "meaningless" in text


def test_summarize_depth_reports_median_distance_and_mean_high_confidence_share() -> None:
    samples = [CenterDepth(2.0, 0.0), None, CenterDepth(2.2, 0.5), CenterDepth(1.8, 1.0)]
    actual = summarize_depth(samples)
    assert actual == DepthSummary(
        frames=4, with_distance=3, median_m=pytest.approx(2.0), high_ratio=pytest.approx(0.5)
    )


def test_summarize_depth_without_confidence_has_no_ratio() -> None:
    actual = summarize_depth([CenterDepth(1.0, None)])
    assert actual == DepthSummary(frames=1, with_distance=1, median_m=pytest.approx(1.0), high_ratio=None)


def test_format_summary_appends_a_depth_line() -> None:
    samples = [
        PoseSample(seq=i, t=i / 30, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))
        for i in range(3)
    ]
    text = format_summary(
        summarize(samples, max_interval_s=0.2, max_speed_mps=2.0),
        depth=DepthSummary(frames=20, with_distance=20, median_m=2.01, high_ratio=0.0),
    )
    assert "depth: 20 frames" in text
    assert "center 2.01 m" in text
    assert "high confidence 0%" in text


def test_format_summary_says_when_no_depth_frame_has_a_distance() -> None:
    samples = [
        PoseSample(seq=i, t=i / 30, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))
        for i in range(3)
    ]
    text = format_summary(
        summarize(samples, max_interval_s=0.2, max_speed_mps=2.0),
        depth=DepthSummary(frames=5, with_distance=0, median_m=None, high_ratio=None),
    )
    assert "depth: 5 frames, no valid distance at the center" in text
