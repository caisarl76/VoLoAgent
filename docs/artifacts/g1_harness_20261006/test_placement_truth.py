"""Acceptance counterexamples for independent simulation placement truth."""

import copy
import math

import pytest

from placement_truth import placement_confirmed


DESTINATION = {"geom": "right_green_stool", "xy": [0.35, -0.6], "radius": 0.25, "height": 0.8}


def samples():
    return [{"at": 10+i/10, "sim_time": 20+i/10,
             "bottle_position": [0.35, -0.6, 0.875], "bottle_bottom_z": 0.8,
             "bottle_xy_radius": 0.03, "bottle_linear_speed": 0.001,
             "bottle_angular_speed": 0.005,
             "contacts": [{"geom": "right_green_stool", "is_robot": False,
                           "position_z": 0.8, "normal_abs_z": 1.0}]} for i in range(6)]


def test_stable_released_bottle_on_stool_top_is_confirmed():
    assert placement_confirmed(samples(), DESTINATION, now=10.5)


@pytest.mark.parametrize("case", ["source", "side", "robot", "edge", "moving", "rotating",
                                  "floating", "missing_geometry", "nonfinite", "stale",
                                  "future", "repeated_wall", "repeated_sim", "slow_physics", "sample_gap", "too_short"])
def test_unsafe_or_unproven_placement_is_rejected(case):
    rows = copy.deepcopy(samples())
    now = 10.5
    for row in rows:
        if case == "source":
            row["contacts"][0]["geom"] = "source_table"
        elif case == "side":
            row["contacts"][0].update(position_z=0.7, normal_abs_z=0.0)
        elif case == "robot":
            # An elbow also invalidates release; checking only hand names misses it.
            row["contacts"].append({"geom": "left_elbow", "is_robot": True,
                                     "position_z": 0.9, "normal_abs_z": 0.0})
        elif case == "edge":
            row["bottle_position"][0] = 0.59
        elif case == "moving":
            row["bottle_linear_speed"] = 0.1
        elif case == "rotating":
            row["bottle_angular_speed"] = 0.3
        elif case == "floating":
            row["bottle_bottom_z"] = 0.95
        elif case == "missing_geometry":
            del row["bottle_bottom_z"]
        elif case == "nonfinite":
            row["bottle_position"][0] = math.nan
    if case == "stale":
        now = 11.0
    elif case == "future":
        now = 10.4
    elif case == "repeated_wall":
        rows[-1]["at"] = rows[-2]["at"]
    elif case == "repeated_sim":
        rows[-1]["sim_time"] = rows[-2]["sim_time"]
    elif case == "slow_physics":
        for i, row in enumerate(rows):
            row["sim_time"] = 20 + i * 0.001
    elif case == "sample_gap":
        rows = [rows[0], rows[-1]]
    elif case == "too_short":
        rows = rows[-3:]
    assert not placement_confirmed(rows, DESTINATION, now=now)


@pytest.mark.parametrize("field", ["xy", "radius", "height"])
def test_nonfinite_destination_cannot_confirm_placement(field):
    destination = copy.deepcopy(DESTINATION)
    destination[field] = [math.nan, -0.6] if field == "xy" else math.nan
    assert not placement_confirmed(samples(), destination, now=10.5)
