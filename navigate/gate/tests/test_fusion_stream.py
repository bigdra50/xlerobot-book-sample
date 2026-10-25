import math

import pytest

from navgate.fusion import FusionParams, Pose2D, fuse_step
from navgate.fusion_stream import FusionStream, body_delta

P = FusionParams()


def test_body_delta_inverts_the_mid_heading_step():
    start = Pose2D(0.3, -0.2, 1.0)
    end = fuse_step(start, 0.05, -0.02, 0.1, None, P).pose
    assert body_delta(start, end) == pytest.approx((0.05, -0.02, 0.1))


def test_the_first_wheel_pose_sets_the_origin():
    fs = FusionStream(P, max_delay_s=0.3)
    fs.feed_wheel(0.0, Pose2D(1.0, 2.0, 0.5))
    assert fs.pose == Pose2D(1.0, 2.0, 0.5)
    assert fs.ready(now=0.0) == []


def test_waits_until_arkit_covers_the_interval():
    fs = FusionStream(P, max_delay_s=0.3)
    fs.feed_arkit(-0.01, 0.0, ok=True, epoch=1)
    fs.feed_wheel(0.0, Pose2D())
    fs.feed_wheel(0.02, Pose2D(0.0, 0.0, 0.01))
    assert fs.ready(now=0.03) == []
    fs.feed_arkit(0.03, 0.0102 * 0.03 / 0.02, ok=True, epoch=1)
    out = fs.ready(now=0.04)
    assert len(out) == 1
    t, pose, weight = out[0]
    assert t == 0.02
    assert weight == 0.0  # その場の旋回なので車輪の値
    assert pose.theta == pytest.approx(0.01)


def test_translation_takes_the_arkit_yaw():
    fs = FusionStream(P, max_delay_s=0.3)
    fs.feed_arkit(-0.01, 0.0, ok=True, epoch=1)
    fs.feed_wheel(0.0, Pose2D())
    fs.feed_wheel(0.1, Pose2D(0.0, 0.01, 0.0))
    fs.feed_arkit(0.11, 0.0022, ok=True, epoch=1)
    ((_, pose, weight),) = fs.ready(now=0.12)
    assert weight == 1.0
    assert pose.theta == pytest.approx(0.0022 * 0.1 / 0.12)  # 補間した yaw の、t=0 から t=0.1 までの変化


def test_without_arkit_the_wheels_are_used_after_the_delay():
    fs = FusionStream(P, max_delay_s=0.3)
    fs.feed_wheel(0.0, Pose2D())
    fs.feed_wheel(0.02, Pose2D(0.0, 0.01, 0.0))
    assert fs.ready(now=0.1) == []
    ((t, pose, weight),) = fs.ready(now=0.33)
    assert (t, weight) == (0.02, 0.0)
    assert pose.y == pytest.approx(0.01)


@pytest.mark.parametrize(("ok", "epochs"), [(False, (1, 1)), (True, (1, 2))])
def test_arkit_is_not_used_while_tracking_is_not_normal_or_across_a_reset(ok, epochs):
    fs = FusionStream(P, max_delay_s=0.3)
    fs.feed_arkit(-0.01, 0.0, ok=ok, epoch=epochs[0])
    fs.feed_wheel(0.0, Pose2D())
    fs.feed_wheel(0.1, Pose2D(0.0, 0.01, 0.0))
    fs.feed_arkit(0.11, 0.5, ok=ok, epoch=epochs[1])
    ((_, pose, weight),) = fs.ready(now=0.12)
    assert weight == 0.0
    assert pose.theta == 0.0


def test_arkit_yaw_across_pi_is_unwrapped():
    fs = FusionStream(P, max_delay_s=0.3)
    fs.feed_arkit(-0.01, math.pi - 0.001, ok=True, epoch=1)
    fs.feed_wheel(0.0, Pose2D())
    fs.feed_wheel(0.1, Pose2D(0.0, 0.01, 0.0))
    fs.feed_arkit(0.11, -math.pi + 0.001, ok=True, epoch=1)  # +0.002 rad で ±π を越えた
    ((_, pose, _),) = fs.ready(now=0.12)
    assert pose.theta == pytest.approx(0.002 * 0.1 / 0.12, abs=1e-6)


def _lens(x, y, th, lens=(0.128, 0.085), mount=0.0524):
    c, s = math.cos(th), math.sin(th)
    return x + c * lens[0] - s * lens[1], y + s * lens[0] + c * lens[1], th + mount


def test_a_push_while_the_wheels_stand_still_moves_the_pose():
    fs = FusionStream(P, max_delay_s=0.3, lens_xy=(0.128, 0.085), mount_yaw=0.0524)
    for i in range(-1, 32):
        t = i / 30
        frac = min(max(t, 0.0), 1.0)
        x, y, a = _lens(0.1 * frac, 0.0, 0.0)  # 1 秒で前へ 0.1 m 押された
        fs.feed_arkit(t, a, ok=True, epoch=1, x=x, y=y)
    for i in range(31):
        fs.feed_wheel(i / 30, Pose2D())
    out = fs.ready(now=1.2)
    assert out[-1][1].x == pytest.approx(0.1, abs=1e-3)
    assert out[-1][1].y == pytest.approx(0.0, abs=1e-3)


def test_without_arkit_positions_a_push_is_not_seen():
    fs = FusionStream(P, max_delay_s=0.3)
    for i in range(-1, 32):
        fs.feed_arkit(i / 30, 0.0, ok=True, epoch=1)
    for i in range(31):
        fs.feed_wheel(i / 30, Pose2D())
    assert fs.ready(now=1.2)[-1][1] == Pose2D()
