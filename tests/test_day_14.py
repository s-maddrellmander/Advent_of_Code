import pytest

from solutions.year_2023.day_14 import *

test_cases = [
    (
        [
            "O....#....",
            "O.OO#....#",
            ".....##...",
            "OO.#O....O",
            ".O.....O#.",
            "O.#..O.#.#",
            "..O..#O..O",
            ".......O..",
            "#....###..",
            "#OO..#....",
        ],
        [
            "OOOO.#.O..",
            "OO..#....#",
            "OO..O##..O",
            "O..#.OO...",
            "........#.",
            "..#....#.#",
            "..O..#.O.O",
            "..O.......",
            "#....###..",
            "#....#....",
        ],
    )
]


@pytest.mark.parametrize("inputs, expected", test_cases)
def test_fast_rolling(inputs, expected):
    inputs = np.array([list(row) for row in inputs])
    inputs = np.rot90(inputs, 2)
    expected = np.array([list(row) for row in expected])
    expected = np.rot90(expected, 2)

    grid_array = roll_fast_with_obstacles(inputs)
    assert np.array_equal(grid_array, expected)

    # Check the score on the test as well
    score = score_grid(grid_array)
    assert score == 136


@pytest.mark.parametrize("inputs, expected", test_cases)
def test_part2(inputs, expected):
    score = part2(inputs)
    assert score == 64
