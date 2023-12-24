import pytest

from solutions.year_2023.day_23 import *

example_map = [
    "#.#####################",
    "#.......#########...###",
    "#######.#########.#.###",
    "###.....#.>.>.###.#.###",
    "###v#####.#v#.###.#.###",
    "###.>...#.#.#.....#...#",
    "###v###.#.#.#########.#",
    "###...#.#.#.......#...#",
    "#####.#.#.#######.#.###",
    "#.....#.#.#.......#...#",
    "#.#####.#.#.#########v#",
    "#.#...#...#...###...>.#",
    "#.#.#v#######v###.###v#",
    "#...#.>.#...>.>.#.###.#",
    "#####v#.#.###v#.#.###.#",
    "#.....#...#...#.#.#...#",
    "#.#########.###.#.#.###",
    "#...###...#...#...#.###",
    "###.###.#.###v#####v###",
    "#...#...#.#.>.>.#.>.###",
    "#.###.###.#.###.#.#v###",
    "#.....###...###...#...#",
    "#####################.#",
]


def test_parse_map():
    result = parse_map(example_map)
    assert result[(1, 0)] == "."


def test_get_neighbors():
    map_data = {
        (0, 0): ".",
        (1, 0): ">",
        (2, 0): ".",
        (0, 1): "^",
        (1, 1): ".",
        (2, 1): "v",
        (0, 2): ".",
        (1, 2): "<",
        (2, 2): ".",
    }

    assert get_neighbors((1, 0), map_data) == [(2, 0)]
    assert get_neighbors((0, 1), map_data) == [(0, 0)]
    assert get_neighbors((2, 1), map_data) == [(2, 2)]
    assert get_neighbors((1, 2), map_data) == [(0, 2)]
    assert set(get_neighbors((1, 1), map_data)) == set([(0, 1), (2, 1), (1, 0), (1, 2)])

    with pytest.raises(KeyError):
        get_neighbors((3, 3), map_data)


def test_part1_step_by_step():
    parsed_map = parse_map(example_map)
    length = longest_path((1, 0), (21, 22), parsed_map)

    assert length == 94


def test_part1():
    assert part1(example_map) == 94


def test_part2():
    assert part2(example_map) == 154


def test_longest_path():
    map_data = {
        (0, 0): ".",
        (1, 0): ".",
        (2, 0): ".",
        (0, 1): ".",
        (1, 1): ".",
        (2, 1): ".",
        (0, 2): ".",
        (1, 2): ".",
        (2, 2): ".",
    }

    assert longest_path((0, 0), (2, 2), map_data) == 8
    assert longest_path((0, 0), (0, 0), map_data) == 0
    assert longest_path((0, 0), (1, 0), map_data) == 7

    with pytest.raises(KeyError):
        longest_path((3, 3), (0, 0), map_data)
