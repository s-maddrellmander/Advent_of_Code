import pytest

from solutions.year_2023.day_22 import *

example_input = [
    "1,0,1~1,2,1",
    "0,0,2~2,0,2",
    "0,2,3~2,2,3",
    "0,0,4~0,2,4",
    "2,0,5~2,2,5",
    "0,1,6~2,1,6",
    "1,1,8~1,1,9",
]


def test_parse_input():
    # Parse the example input
    parsed_bricks = parse_input(example_input)
    assert parsed_bricks[0] == ((1, 0, 1), (1, 2, 1))
    assert len(parsed_bricks) == 7
    assert parsed_bricks[-1] == ((1, 1, 8), (1, 1, 9))


def test_tuples_to_bricks():
    parsed_bricks = parse_input(example_input)
    # Convert the parsed bricks into a list of bricks
    bricks = tuples_to_bricks(parsed_bricks)
    assert bricks[0].id == 0
    assert bricks[0].cells == ((1, 0, 1), (1, 1, 1), (1, 2, 1))
    assert bricks[0].minZ == 1
    assert bricks[-1].id == 6
    assert bricks[-1].cells == ((1, 1, 8), (1, 1, 9))
    assert bricks[-1].minZ == 8


def test_lower_brick():
    parsed_bricks = parse_input(example_input)
    bricks = tuples_to_bricks(parsed_bricks)
    # Lower the first brick
    lower_brick_0 = lower_brick(bricks[0])
    assert lower_brick_0.id == 0
    assert lower_brick_0.cells == ((1, 0, 0), (1, 1, 0), (1, 2, 0))
    assert lower_brick_0.minZ == 0


def test_simulate_settling():
    brick1 = Brick(1, ((0, 0, 0), (1, 0, 0), (2, 0, 0)), 0)
    brick2 = Brick(2, ((0, 1, 2), (1, 1, 2), (2, 1, 2)), 2)
    brick3 = Brick(3, ((0, 1, 4), (1, 1, 4), (2, 1, 4)), 4)
    bricks = [brick1, brick2, brick3]

    result = simulate_settling(bricks)
    expected = {
        1: Brick(1, ((0, 0, 0), (1, 0, 0), (2, 0, 0)), 0),
        2: Brick(2, ((0, 1, 0), (1, 1, 0), (2, 1, 0)), 0),
        3: Brick(3, ((0, 1, 1), (1, 1, 1), (2, 1, 1)), 1),
    }

    assert result == expected


def test_part1():
    assert part1(example_input) == 5


def test_part2():
    assert part2(example_input) == 7
