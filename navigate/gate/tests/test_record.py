"""gate-record の状態行（純粋）と FakeDevice を使った記録の往復。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pocketsensor.testing.fake_device import FakeDevice
from pocketsensor.types import TrackingReason, TrackingState

from navgate.analysis import main as analyze_main
from navgate.analysis import summarize
from navgate.record import StatusSnapshot, format_status_line
from navgate.record import main as record_main
from navgate.recording import read_mcap


def test_format_status_line_includes_elapsed_poses_fps_pose_tracking_depth_clock() -> None:
    line = format_status_line(
        StatusSnapshot(
            elapsed_s=1.5,
            poses=42,
            fps=28.0,
            position=(0.12, -0.03, 0.01),
            tracking_state=int(TrackingState.NORMAL),
            tracking_reason=int(TrackingReason.NONE),
            origin_epoch=0,
            depth_center_m=1.23,
            clock_offset_ms=1.5,
            clock_rtt_ms=4.0,
        )
    )
    assert "1.5" in line
    assert "42" in line
    assert "28.0" in line
    assert "0.12" in line
    assert "normal" in line
    assert "origin" in line
    assert "1.23" in line
    assert "1.5" in line
    assert "4.0" in line


def test_format_status_line_shows_high_confidence_share_next_to_depth() -> None:
    line = format_status_line(
        StatusSnapshot(
            elapsed_s=1.0,
            poses=30,
            fps=30.0,
            position=(0.0, 0.0, 0.0),
            tracking_state=int(TrackingState.NORMAL),
            tracking_reason=int(TrackingReason.NONE),
            origin_epoch=0,
            depth_center_m=2.01,
            clock_offset_ms=0.0,
            clock_rtt_ms=5.0,
            depth_high_ratio=0.0,
        )
    )
    assert "2.01 m" in line
    assert "high conf 0%" in line


def test_format_status_line_marks_missing_optional_fields() -> None:
    line = format_status_line(
        StatusSnapshot(
            elapsed_s=0.0,
            poses=0,
            fps=0.0,
            position=None,
            tracking_state=None,
            tracking_reason=None,
            origin_epoch=None,
            depth_center_m=None,
            clock_offset_ms=None,
            clock_rtt_ms=None,
        )
    )
    assert "n/a" in line


def test_gate_record_fake_device_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """FakeDevice へ ~1.5 s 記録し、MCAP を read_mcap / gate-analyze で読む。"""
    out = tmp_path / "run.mcap"
    # 端末名は既定の pocketsensor にしない。トピック接頭辞の直書きを検出するため。
    with FakeDevice(name="cartcam", port=0, seed=0) as fake:
        code = record_main(["--source", fake.url, "--out", str(out), "--duration", "1.5"])
    assert code == 0
    assert out.is_file()

    recording = read_mcap(out)
    assert len(recording.poses) > 20
    assert recording.tracking
    summary = summarize(recording.poses)
    assert math.isfinite(summary.path_length_m)
    assert math.isfinite(summary.closure_error_m)
    assert math.isfinite(summary.duration_s)

    assert analyze_main([str(out)]) == 0
    captured = capsys.readouterr()
    assert "closure error" in captured.out


def test_gate_record_connection_failure_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = record_main(
        ["--source", "ws://127.0.0.1:1", "--duration", "0.1", "--out", str(tmp_path / "unused.mcap")]
    )
    assert code == 1
    assert capsys.readouterr().err
