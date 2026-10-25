# 出典: NektoDron/lekiwi-vr-teleop 4e1350c（2026-08-04） lekiwi_vr_teleop/so101_chain.py
# Apache-2.0。Copyright 2026 Andrey Demidov。plugin 直下の LICENSE-lekiwi を参照。
# 変更点: なし

"""Closed-form kinematics for the SO-101 arm.

Why not just use the URDF through placo, as the rest of this project does? Because the
SO-101's geometry is not generic, and pretending it is was what made the arm feel
arbitrary to drive.

Measured from ``so101_new_calib.urdf`` (see ``tests/test_so101_chain.py``, which checks
every claim below against placo's own forward kinematics to within 0.01 deg / 0.1 mm):

* ``shoulder_pan`` turns about the base vertical, through the line ``x = PAN_AXIS_X``.
  It selects a vertical *plane*; it does nothing else.
* ``shoulder_lift``, ``elbow_flex`` and ``wrist_flex`` are three parallel revolutes lying
  in that plane — a planar 3R.
* ``wrist_roll`` turns about the tool's own approach axis.

Two exact identities follow, and they are the whole design:

    pan        = -atan2(y, x - PAN_AXIS_X)
    tool pitch = -(shoulder_lift + elbow_flex + wrist_flex)

So the reachable orientation set is not "most of SO(3) with a bit missing". It is exactly
two-dimensional: **pitch and roll are free, yaw is dictated by where the tool is**. An
operator turning their hand about the vertical is asking for something the machine cannot
do in any configuration, and no solver weighting changes that — it only decides which
joints get sacrificed while trying.

Given that, inverse kinematics is a textbook 2R problem and needs no solver: place the
wrist pivot by walking back along the approach axis, solve two links to reach it, and read
the last joint off the pitch identity. One branch, chosen explicitly, so the elbow never
flips mid-motion; and it is exact rather than an iterate, so the joint targets shown in the
headset are the joint targets that get sent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ------------------------------------------------------------------ geometry
# All lengths in metres, all angles in degrees (the SO-101 bus works in degrees).

# The pan axis is offset from the URDF origin: the shoulder does not sit over base_link.
PAN_AXIS_X = 0.0388353

# The planar 3R, in the (r, z) plane of the arm — r measured from the pan axis outward.
SHOULDER_R = 0.030399
SHOULDER_Z = 0.1166

# Link lengths between consecutive pivots, and the fixed mounting angle of each link at
# zero joint angle (the links are not collinear with their pivots; these constants absorb
# that so the joint angles below are true SO-101 joint angles, not model angles).
UPPER_ARM_LEN = 0.116000
UPPER_ARM_ANGLE = 76.0323
FOREARM_LEN = 0.135000
FOREARM_ANGLE = 2.2075

# Wrist pivot -> control point, measured along the approach axis.
#
# The control point is deliberately NOT the URDF's ``gripper_frame_link``. That frame sits
# 7.9 mm off the roll axis, so rolling the wrist swings it around a small circle: the jaw
# would translate sideways every time the operator turned their hand, which is both a
# surprise to drive and a coupling the arm does not actually have. Putting the control
# point on the roll axis makes position and roll genuinely independent.
TOOL_ALONG = 0.159227

# That circle, kept only so the URDF frame can be reconstructed exactly and compared
# against placo. The phase is not 180 deg because the gripper is mounted with a small tilt
# (the 0.0486795 rad in the wrist_roll joint origin). The axis is also a fifth of a
# millimetre off the arm's plane, which is ignored everywhere except here.
JAW_ECCENTRICITY = 0.007903
JAW_ROLL_PHASE_DEG = 178.7923
ROLL_AXIS_LATERAL = -0.000176

# Joint travel, from the URDF <limit> tags, in degrees.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}

REACH_MIN = abs(UPPER_ARM_LEN - FOREARM_LEN)
REACH_MAX = UPPER_ARM_LEN + FOREARM_LEN


@dataclass
class ArmPose:
    """One solved arm configuration, plus how much of the request had to be given up.

    ``position_error_m`` and ``pitch_error_deg`` are not diagnostics after the fact: they
    are what the headset draws so the operator can see the arm refusing a request instead
    of guessing why it stopped following.
    """

    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    position_error_m: float = 0.0
    pitch_error_deg: float = 0.0
    clamped: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "shoulder_pan": self.shoulder_pan,
            "shoulder_lift": self.shoulder_lift,
            "elbow_flex": self.elbow_flex,
            "wrist_flex": self.wrist_flex,
            "wrist_roll": self.wrist_roll,
        }


def tool_pitch(shoulder_lift: float, elbow_flex: float, wrist_flex: float) -> float:
    """Elevation of the tool's approach axis, in degrees. Exact, not an approximation."""
    return -(shoulder_lift + elbow_flex + wrist_flex)


def _clamp(value: float, name: str, current: float | None = None) -> tuple[float, bool]:
    """Clamp to the joint's travel, widened to include where the joint already is.

    The limits here come from the URDF, which describes the *model*. The robot's own
    calibration is what decides what it reports and accepts, and it is routinely wider:
    LeRobot's DEGREES mode returns ``(raw - calibrated_middle) * 360 / 4095`` without
    bounding the result, so a real SO-101 reports angles outside the URDF's range as a
    matter of course — one measured here sat 12 deg past the model's wrist_roll limit.

    Clamping such a joint would command it *away* from where it is standing, which turns
    engaging the clutch into a jerk — breaking the one promise this whole design rests on.
    Since a joint cannot become more illegal by staying put, the limit always yields to the
    measured position, and only ever prevents travelling further out.
    """
    low, high = JOINT_LIMITS[name]
    if current is not None:
        low, high = min(low, current), max(high, current)
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def forward(joints: dict[str, float]) -> dict[str, tuple[float, float, float]]:
    """Pivot positions in the base frame, shoulder to tool, for drawing the arm.

    Returns the four points a stick figure needs plus the tool's approach direction, which
    is what makes a bare skeleton readable — without it you cannot tell which way the jaw
    is facing.
    """
    pan = math.radians(-joints["shoulder_pan"])
    lift = joints["shoulder_lift"]
    elbow = joints["elbow_flex"]
    flex = joints["wrist_flex"]

    a1 = math.radians(UPPER_ARM_ANGLE - lift)
    a2 = math.radians(FOREARM_ANGLE - lift - elbow)
    pitch = math.radians(tool_pitch(lift, elbow, flex))

    shoulder = (SHOULDER_R, SHOULDER_Z)
    elbow_pt = (shoulder[0] + UPPER_ARM_LEN * math.cos(a1), shoulder[1] + UPPER_ARM_LEN * math.sin(a1))
    wrist_pt = (elbow_pt[0] + FOREARM_LEN * math.cos(a2), elbow_pt[1] + FOREARM_LEN * math.sin(a2))
    cp, sp = math.cos(pitch), math.sin(pitch)
    tool_pt = (wrist_pt[0] + TOOL_ALONG * cp, wrist_pt[1] + TOOL_ALONG * sp)

    def to_base(point: tuple[float, float]) -> tuple[float, float, float]:
        r, z = point
        return (PAN_AXIS_X + r * math.cos(pan), r * math.sin(pan), z)

    approach = (cp * math.cos(pan), cp * math.sin(pan), sp)
    # In-plane normal to the approach axis; the jaw frame orbits the roll axis in the
    # plane it spans with the out-of-plane direction.
    normal = (-sp * math.cos(pan), -sp * math.sin(pan), cp)
    lateral = (-math.sin(pan), math.cos(pan), 0.0)
    roll = math.radians(joints.get("wrist_roll", 0.0) + JAW_ROLL_PHASE_DEG)
    tool = to_base(tool_pt)
    return {
        "base": (PAN_AXIS_X, 0.0, 0.0),
        "shoulder": to_base(shoulder),
        "elbow": to_base(elbow_pt),
        "wrist": to_base(wrist_pt),
        "tool": tool,
        # Unit approach direction, so the headset can draw which way the jaw points.
        "approach": approach,
        # The jaw's own frame, off the roll axis — this is the URDF's gripper_frame_link,
        # kept so the model can be checked against placo and so the ghost can show roll.
        "jaw": tuple(
            tool[i]
            + ROLL_AXIS_LATERAL * lateral[i]
            + JAW_ECCENTRICITY * (math.cos(roll) * normal[i] + math.sin(roll) * lateral[i])
            for i in range(3)
        ),
    }


def inverse(
    position: tuple[float, float, float],
    pitch_deg: float,
    roll_deg: float,
    elbow_up: bool = True,
    current: dict[str, float] | None = None,
) -> ArmPose:
    """Solve the arm for a tool point and a tool pitch and roll. Never raises.

    An unreachable request is *clamped to the nearest reachable configuration* rather than
    rejected, and the shortfall is reported on the pose. A teleoperation loop that raises
    when the operator reaches too far leaves the arm uncommanded mid-motion, which is the
    one thing it must not do; and silently freezing is just as bad, because the operator
    cannot tell a dropped connection from a workspace edge.

    The radial coordinate is taken non-negative, so a target *behind* the pan axis is
    solved by reaching over the axis rather than by spinning the base 180 deg to face
    backwards. Both are valid inverses; this one is continuous, and the other is usually
    outside the +-110 deg pan limit anyway. It does mean targets within a few centimetres
    of the axis are ill-conditioned — a small sideways move swings the pan a long way —
    which is a property of the machine, not of this solver.
    """
    current = current or {}
    x, y, z = position
    dx = x - PAN_AXIS_X
    pan_raw = -math.degrees(math.atan2(y, dx))
    pan, pan_clamped = _clamp(pan_raw, "shoulder_pan", current.get("shoulder_pan"))

    # Radial distance, signed by the pan clamp: if the pan could not turn far enough, the
    # tool cannot be where it was asked to be, and the radial coordinate is measured along
    # the plane the arm actually reached.
    if pan_clamped:
        theta = math.radians(-pan)
        r = dx * math.cos(theta) + y * math.sin(theta)
    else:
        r = math.hypot(dx, y)

    # Walk back from the tool point along the approach axis to the wrist pivot.
    pitch = math.radians(pitch_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    wr = r - TOOL_ALONG * cp
    wz = z - TOOL_ALONG * sp

    # Two-link reach to the wrist pivot.
    ur, uz = wr - SHOULDER_R, wz - SHOULDER_Z
    reach = math.hypot(ur, uz)
    clipped = min(max(reach, REACH_MIN + 1e-6), REACH_MAX - 1e-6)
    reach_clamped = abs(clipped - reach) > 1e-9
    if reach > 1e-9:
        ur, uz = ur * clipped / reach, uz * clipped / reach
    else:
        # Degenerate: the wrist pivot is on the shoulder. Push it straight out so the
        # solution stays continuous instead of jumping to an arbitrary angle.
        ur, uz = clipped, 0.0
    reach = clipped

    cos_inner = (reach * reach - UPPER_ARM_LEN**2 - FOREARM_LEN**2) / (2 * UPPER_ARM_LEN * FOREARM_LEN)
    inner = math.acos(min(1.0, max(-1.0, cos_inner)))
    # Angle between the upper arm and the straight line to the wrist pivot.
    offset = math.atan2(FOREARM_LEN * math.sin(inner), UPPER_ARM_LEN + FOREARM_LEN * math.cos(inner))
    direct = math.atan2(uz, ur)
    sign = 1.0 if elbow_up else -1.0
    upper_abs = direct + sign * offset
    fore_abs = upper_abs - sign * inner

    lift_raw = UPPER_ARM_ANGLE - math.degrees(upper_abs)
    elbow_raw = FOREARM_ANGLE - math.degrees(fore_abs) - lift_raw
    # The pitch identity gives the last joint directly — no third IK stage.
    flex_raw = -pitch_deg - lift_raw - elbow_raw

    lift, c1 = _clamp(lift_raw, "shoulder_lift", current.get("shoulder_lift"))
    elbow, c2 = _clamp(elbow_raw, "elbow_flex", current.get("elbow_flex"))
    flex, c3 = _clamp(flex_raw, "wrist_flex", current.get("wrist_flex"))
    roll, c4 = _clamp(roll_deg, "wrist_roll", current.get("wrist_roll"))

    pose = ArmPose(
        shoulder_pan=pan,
        shoulder_lift=lift,
        elbow_flex=elbow,
        wrist_flex=flex,
        wrist_roll=roll,
        clamped=pan_clamped or reach_clamped or c1 or c2 or c3 or c4,
    )
    # Report the shortfall by measuring the solution we actually produced, so clamping at
    # any stage — pan, reach or a joint limit — shows up in one number.
    reached = forward(pose.as_dict())["tool"]
    pose.position_error_m = math.dist(reached, position)
    pose.pitch_error_deg = tool_pitch(lift, elbow, flex) - pitch_deg
    return pose
