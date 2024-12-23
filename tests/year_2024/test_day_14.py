import pytest

from solutions.year_2024.day_14 import *


def test_single_point_modulo():
    "p=2,4 v=2,-3"
    x = 2
    y = 4
    t = 1

    x_t = x + 2 * t
    x_t = x_t % 11
    assert x_t == 4

    y_t = y - 3 * t
    y_t = y_t % 7
    assert y_t == 1

    t = 3
    x_t = x + 2 * t
    x_t = x_t % 11
    assert x_t == 8

    y_t = y - 3 * t
    y_t = y_t % 7
    assert y_t == 2

    t = 5
    x_t = x + 2 * t
    x_t = x_t % 11
    assert x_t == 1
    y_t = y - 3 * t
    y_t = y_t % 7
    assert y_t == 3


@pytest.fixture
def test_data():
    return [
        "p=0,4 v=3,-3",
        "p=6,3 v=-1,-3",
        "p=10,3 v=-1,2",
        "p=2,0 v=2,-1",
        "p=0,0 v=1,3",
        "p=3,0 v=-2,-2",
        "p=7,6 v=-1,-3",
        "p=3,0 v=-1,-2",
        "p=9,3 v=2,3",
        "p=7,3 v=-1,2",
        "p=2,4 v=2,-3",
        "p=9,5 v=-3,-3",
    ]


def test_parse_data(test_data):
    guards = parse_data(test_data)
    assert guards[0] == (complex(0, 4), complex(3, -3))


def test_move_guard():
    guard = (complex(2, 4), complex(2, -3))
    bounds = (11, 7)
    assert move_guard(guard, 1, bounds) == (4, 1)
    assert move_guard(guard, 3, bounds) == (8, 2)
    assert move_guard(guard, 5, bounds) == (1, 3)


def test_all_guards_100s(test_data):
    guards = parse_data(test_data)
    bounds = (11, 7)
    # For each coordinate in the grid, count the gaurds in the cell
    grid = [[0 for _ in range(11)] for _ in range(7)]
    for guard in guards:
        x, y = move_guard(guard, 100, bounds)
        grid[y][x] += 1
    assert sum(grid[0]) == 3
    assert grid[0][6] == 2
    assert grid[0][9] == 1

    assert grid[2][0] == 1
    assert sum(grid[2]) == 1
    assert sum(grid[3]) == 2
    assert grid[3][2] == 1
    assert grid[3][1] == 1
    assert sum(grid[6]) == 2
    assert grid[6][1] == 1
    assert grid[5][3] == 1
    assert grid[5][4] == 2


def test_put_in_quadrant():
    bounds = (11, 7)

    probe = (0, 2)
    assert put_in_quadrant(probe, bounds) == 0
    probe = (6, 0)
    assert put_in_quadrant(probe, bounds) == 1

    probe = (2, 3)
    assert put_in_quadrant(probe, bounds) == -1


def test_part1(test_data):
    assert part1(test_data, bounds=(11, 7)) == 12
