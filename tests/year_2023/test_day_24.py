import pytest

from solutions.year_2023.day_24 import *

input_data = [
    "19, 13, 30 @ -2,  1, -2",
    "18, 19, 22 @ -1, -1, -2",
    "20, 25, 34 @ -2, -2, -4",
    "12, 31, 28 @ -1, -2, -1",
    "20, 19, 15 @  1, -5, -3",
]


def test_parse_data():
    coord_velocities = parse_data(input_data)
    expected = [
        [19, 13, 30, -2, 1, -2],
        [18, 19, 22, -1, -1, -2],
        [20, 25, 34, -2, -2, -4],
        [12, 31, 28, -1, -2, -1],
        [20, 19, 15, 1, -5, -3],
    ]
    assert coord_velocities == expected


def test_crossing_paths():
    # Hailstone A: 19, 13, 30 @ -2, 1, -2
    # Hailstone B: 18, 19, 22 @ -1, -1, -2
    # Hailstones' paths will cross inside the test area (at x=14.333, y=15.333).
    first_two = [19, 13, 30, -2, 1, -2], [18, 19, 22, -1, -1, -2]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == [(14, 15)]
    # Hailstone A: 19, 13, 30 @ -2, 1, -2
    # Hailstone B: 20, 25, 34 @ -2, -2, -4
    # Hailstones' paths will cross inside the test area (at x=11.667, y=16.667).
    first_two = [19, 13, 30, -2, 1, -2], [20, 25, 34, -2, -2, -4]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == [(11, 16)]

    # Hailstone A: 19, 13, 30 @ -2, 1, -2
    # Hailstone B: 12, 31, 28 @ -1, -2, -1
    # Hailstones' paths will cross outside the test area (at x=6.2, y=19.4).
    first_two = [19, 13, 30, -2, 1, -2], [12, 31, 28, -1, -2, -1]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 19, 13, 30 @ -2, 1, -2
    # Hailstone B: 20, 19, 15 @ 1, -5, -3
    # Hailstones' paths crossed in the past for hailstone A.
    first_two = [19, 13, 30, -2, 1, -2], [20, 19, 15, 1, -5, -3]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 18, 19, 22 @ -1, -1, -2
    # Hailstone B: 20, 25, 34 @ -2, -2, -4
    # Hailstones' paths are parallel; they never intersect.
    first_two = [18, 19, 22, -1, -1, -2], [20, 25, 34, -2, -2, -4]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 18, 19, 22 @ -1, -1, -2
    # Hailstone B: 12, 31, 28 @ -1, -2, -1
    # Hailstones' paths will cross outside the test area (at x=-6, y=-5).
    first_two = [18, 19, 22, -1, -1, -2], [12, 31, 28, -1, -2, -1]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 18, 19, 22 @ -1, -1, -2
    # Hailstone B: 20, 19, 15 @ 1, -5, -3
    # Hailstones' paths crossed in the past for both hailstones.
    first_two = [18, 19, 22, -1, -1, -2], [20, 19, 15, 1, -5, -3]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 20, 25, 34 @ -2, -2, -4
    # Hailstone B: 12, 31, 28 @ -1, -2, -1
    # Hailstones' paths will cross outside the test area (at x=-2, y=3).
    first_two = [20, 25, 34, -2, -2, -4], [12, 31, 28, -1, -2, -1]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 20, 25, 34 @ -2, -2, -4
    # Hailstone B: 20, 19, 15 @ 1, -5, -3
    # Hailstones' paths crossed in the past for hailstone B.
    first_two = [20, 25, 34, -2, -2, -4], [20, 19, 15, 1, -5, -3]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []

    # Hailstone A: 12, 31, 28 @ -1, -2, -1
    # Hailstone B: 20, 19, 15 @ 1, -5, -3
    # Hailstones' paths crossed in the past for both hailstones.
    first_two = [12, 31, 28, -1, -2, -1], [20, 19, 15, 1, -5, -3]
    intersect = crossing_paths(first_two, 7, 27)
    assert intersect == []


def test_part1():
    assert part1(input_data, min_val=7, max_val=27) == 2


def test_part2():
    assert part2(input_data) == 47
