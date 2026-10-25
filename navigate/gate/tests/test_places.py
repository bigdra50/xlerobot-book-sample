import json

import pytest

from navgate.places import Place, Step, load_places, next_step, relative, save_place

DESK = Place(name="desk", pose=(0.0, 0.0, 0.0), entry=(-0.3, 1.0, 0.0), standoff=0.15)


def test_places_round_trip_through_json(tmp_path):
    path = tmp_path / "places.json"
    save_place(path, DESK)
    assert json.loads(path.read_text())["desk"]["standoff"] == 0.15
    assert load_places(path) == {"desk": DESK}


def test_saving_keeps_other_places(tmp_path):
    path = tmp_path / "places.json"
    save_place(path, DESK)
    save_place(path, Place(name="door", pose=(1.0, 2.0, 90.0), entry=(1.0, 1.5, 90.0), standoff=0.1))
    assert set(load_places(path)) == {"desk", "door"}


def test_relative_pose_is_expressed_in_the_place_frame():
    place = Place(name="p", pose=(1.0, 1.0, 90.0), entry=(0, 0, 0), standoff=0.1)
    long, lat, dyaw = relative((1.0, 0.8, 80.0), place)
    assert long == pytest.approx(-0.2)  # 20 cm behind along the place heading (+y)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert dyaw == pytest.approx(-10.0)


def test_the_heading_is_fixed_first():
    assert next_step((0.0, 0.5, 5.0), DESK) == Step("turn", -5.0)


def test_when_off_to_the_side_it_backs_off_to_the_standoff_before_sliding():
    step = next_step((-0.05, 0.5, 0.0), DESK)
    assert step.kind == "fwd" and step.amount == pytest.approx(-0.10)


def test_sliding_sideways_is_done_in_limited_chunks():
    step = next_step((-0.15, 0.9, 0.0), DESK)
    assert step == Step("left", -0.25)


def test_the_last_move_is_straight_forward_to_the_place():
    step = next_step((-0.15, 0.003, 0.0), DESK)
    assert step.kind == "fwd" and step.amount == pytest.approx(0.15)


def test_nothing_to_do_at_the_place():
    assert next_step((0.002, -0.004, 0.5), DESK) is None


def test_a_place_too_far_to_reach_without_nav2_is_refused():
    with pytest.raises(ValueError):
        next_step((-1.5, 0.0, 0.0), DESK)


def test_leaving_a_place_walks_the_same_steps_back_to_its_entry():
    from navgate.places import exit_of

    out = exit_of(DESK)
    assert out.pose == DESK.entry and out.standoff == DESK.standoff
    # 机の前（原点）から待機位置へは、まず下がる
    step = next_step((0.0, 0.0, 0.0), out)
    assert step.kind == "fwd" and step.amount < 0


def test_near_tells_whether_the_robot_is_at_a_place():
    from navgate.places import near

    assert near((0.05, -0.1, 0.0), DESK, radius=0.3)
    assert not near((0.5, 0.5, 0.0), DESK, radius=0.3)


def test_a_marker_reference_round_trips_and_is_optional(tmp_path):
    from navgate.marker_align import MarkerRef

    path = tmp_path / "places.json"
    ref = MarkerRef(marker_id=48, marker_xy=(0.558, -0.02), wall_yaw=-0.0866)
    save_place(path, Place("desk", (0.0, 0.0, 0.0), (-0.3, 1.0, 0.0), 0.15, marker=ref))
    save_place(path, Place("door", (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), 0.15))
    places = load_places(path)
    assert places["desk"].marker == ref
    assert places["door"].marker is None
    assert "marker" not in json.loads(path.read_text())["door"]
