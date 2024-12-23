import pytest

from solutions.year_2024.day_16 import *


@pytest.fixture
def example_map():
    return [
        "###############",
        "#.......#....E#",
        "#.#.###.#.###.#",
        "#.....#.#...#.#",
        "#.###.#####.#.#",
        "#.#.#.......#.#",
        "#.#.#####.###.#",
        "#...........#.#",
        "###.#.#####.#.#",
        "#...#.....#.#.#",
        "#.#.#.###.#.#.#",
        "#.....#...#.#.#",
        "#.###.#.#.#.#.#",
        "#S..#.....#...#",
        "###############",
    ]


@pytest.fixture
def second_map():
    return [
        "#################",
        "#...#...#...#..E#",
        "#.#.#.#.#.#.#.#.#",
        "#.#.#.#...#...#.#",
        "#.#.#.#.###.#.#.#",
        "#...#.#.#.....#.#",
        "#.#.#.#.#.#####.#",
        "#.#...#.#.#.....#",
        "#.#.#####.#.###.#",
        "#.#.#.......#...#",
        "#.#.###.#####.###",
        "#.#.#...#.....#.#",
        "#.#.#.#####.###.#",
        "#.#.#.........#.#",
        "#.#.#.#########.#",
        "#S#.............#",
        "#################",
    ]


def test_map_to_graph(example_map):
    graph, s, e = map_to_graph(example_map)
    print(graph)
    assert graph[complex(1, 1)] == [complex(1, 2), complex(2, 1)]
    assert s == complex(1, 13)
    assert e == complex(13, 1)


def test_dikstra(example_map):
    graph, s, e = map_to_graph(example_map)
    assert dikstra(graph, s, e) == 7036


def test_dikstra_second(second_map):
    graph, s, e = map_to_graph(second_map)
    assert dikstra(graph, s, e) == 11048


def test_dikstra_second(second_map):
    graph, s, e = map_to_graph(second_map)
    assert dikstra_paths(graph, s, e) == 64
