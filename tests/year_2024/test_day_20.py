import pytest

from solutions.year_2024.day_20 import *
from collections import Counter


@pytest.fixture
def input_data():
    return [
        "###############",
        "#...#...#.....#",
        "#.#.#.#.#.###.#",
        "#S#...#.#.#...#",
        "#######.#.#.###",
        "#######.#.#...#",
        "#######.#.###.#",
        "###..E#...#...#",
        "###.#######.###",
        "#...###...#...#",
        "#.#####.#.###.#",
        "#.#...#.#.#...#",
        "#.#.#.#.#.#.###",
        "#...#...#...###",
        "###############",
    ]


def test_parse_map(input_data):
    carte, s, e = parse_map(input_data)
    print_map(carte)
    assert len(carte) == 85


def test_part1(input_data):
    assert part1(input_data) == 72
    
def test_part1_manual(input_data):
    carte, start, end = parse_map(input_data)
    # Start by finding the baseline path 
    came_from, cost_so_far = a_star(carte, start, end)
    path = reconstruct_path(came_from, start, end)
    print(len(path))
    print_map(carte, path, limit=len(input_data))
    shortcuts = shortcut(path)
    # Count the shortcuts on the savings
    print(Counter(shortcuts))
    
    assert len(shortcuts) == 44