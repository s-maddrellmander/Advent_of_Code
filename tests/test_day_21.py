import pytest

from solutions.year_2023.day_21 import *

SIMPLE_GRID = [
    "...........",
    ".....###.#.",
    ".###.##..#.",
    "..#.#...#..",
    "....#.#....",
    ".##..S####.",
    ".##..#...#.",
    ".......##..",
    ".##.#.####.",
    ".##..##.##.",
    "...........",
]


def test_parse_input():
    grid = parse_input(SIMPLE_GRID)
    assert grid[complex(5, 5)].value == "S"
    assert grid[complex(5, 5)].state == True
    assert grid[complex(6, 5)].value == "#"
    assert grid[complex(0, 0)].value == "."
    print_grid(grid)


def test_get_neighbours():
    # Define a simple grid
    grid = {
        0 + 0j: "Location1",
        1 + 0j: "Location2",
        0 + 1j: "Location3",
        1 + 1j: "Location4",
    }

    # Test the function with a location in the grid
    neighbours = get_neighbours(grid, 0 + 0j)
    assert set(neighbours) == set([1 + 0j, 0 + 1j])

    # Test the function with a location at the edge of the grid
    neighbours = get_neighbours(grid, 1 + 0j)
    assert set(neighbours) == set([0 + 0j, 1 + 1j])


def test_turn_all_states_false():
    grid = parse_input(SIMPLE_GRID)
    grid = turn_all_states_false(grid)
    assert grid[complex(5, 5)].state == False
    assert grid[complex(6, 5)].state == False
    assert grid[complex(0, 0)].state == False


def test_step_all():
    grid = parse_input(SIMPLE_GRID)
    grid = step_all(grid)
    print_grid(grid)
    grid = step_all(grid)
    print_grid(grid)


def test_simple_step():
    _grid = [
        ".....",
        ".O.O.",
        ".....",
    ]

    grid = parse_input(_grid)
    assert grid[complex(1, 1)].value == "O"
    neighs = get_neighbours(grid, complex(1, 1))
    assert set(neighs) == set(
        [complex(1, 0), complex(2, 1), complex(1, 2), complex(0, 1)]
    )
    new_grid = step_all(grid)
    assert count_O(new_grid) == 7
    new_grid = step_all(new_grid)
    assert count_O(new_grid) == 8


def test_basic_grid():
    grid = parse_input(SIMPLE_GRID)
    grid = step_all(grid)
    assert count_O(grid) == 2
    grid = step_all(grid)
    assert count_O(grid) == 4
    grid = step_all(grid)
    assert count_O(grid) == 6
    grid = step_all(grid)
    grid = step_all(grid)
    grid = step_all(grid)
    assert count_O(grid) == 16


def test_part1():
    assert part1(SIMPLE_GRID) == 42
