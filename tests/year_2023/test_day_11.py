import pytest

from solutions.year_2023.day_11 import *


def test_parser():
    input_data = [
        "...#......",
        ".......#..",
        "#.........",
        "..........",
        "......#...",
        ".#........",
        ".........#",
        "..........",
        ".......#..",
        "#...#.....",
    ]
    galaxies = parse_input(input_data)
    assert len(galaxies) == len(input_data) + 2
    assert len(galaxies[0]) == len(input_data[0]) + 3


def test_grid_to_coords():
    input_data = [
        "...#......",
        ".......#..",
        "#.........",
        "..........",
        "......#...",
        ".#........",
        ".........#",
        "..........",
        ".......#..",
        "#...#.....",
    ]
    galaxies = parse_input(input_data)
    coords = grid_to_coords(galaxies)
    list_of_coords = [(0, 4), (11, 5)]
    for coord in list_of_coords:
        assert coord in coords


def test_all_combinations():
    input_data = [
        "...#......",
        ".......#..",
        "#.........",
        "..........",
        "......#...",
        ".#........",
        ".........#",
        "..........",
        ".......#..",
        "#...#.....",
    ]
    galaxies = parse_input(input_data)
    coords = grid_to_coords(galaxies)
    all_pairs = all_combinations(coords)
    assert len(all_pairs) == 36


def test_part1():
    input_data = [
        "...#......",
        ".......#..",
        "#.........",
        "..........",
        "......#...",
        ".#........",
        ".........#",
        "..........",
        ".......#..",
        "#...#.....",
    ]
    output = part1(input_data=input_data)
    assert output == 374


@pytest.mark.parametrize("expansion,expected", [(2, 374), (10, 1030), (100, 8410)])
def test_part2(expansion, expected):
    input_data = [
        "...#......",
        ".......#..",
        "#.........",
        "..........",
        "......#...",
        ".#........",
        ".........#",
        "..........",
        ".......#..",
        "#...#.....",
    ]
    galaxies = parse_input(input_data, expansion_constant=expansion)
    coords = grid_to_coords(galaxies)
    all_pairs = all_combinations(coords)
    distances = manhattan_distances(all_pairs)
    assert expected == sum(distances)


@pytest.mark.parametrize("expansion,expected", [(2, 374), (10, 1030), (100, 8410)])
def test_part2_v2(expansion, expected):
    input_data = [
        "...#......",
        ".......#..",
        "#.........",
        "..........",
        "......#...",
        ".#........",
        ".........#",
        "..........",
        ".......#..",
        "#...#.....",
    ]
    galaxies = parse_input_part2(input_data, expansion_constant=expansion)
    all_pairs = all_combinations(galaxies)
    distances = manhattan_distances(all_pairs)
    assert sum(distances) == expected
