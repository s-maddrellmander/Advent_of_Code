import pytest

from solutions.year_2023.day_17 import *


@pytest.mark.skip
def test_simplified():
    data = [
        "2413",
        "3211",
        "3255",
        "3446",
    ]
    mapping = parser(data)
    print_grid_and_path(mapping, path=[])
    path = dijkstra(mapping, (0, 0), (3, 3), step_limit=85)
    print_grid_and_path(mapping, path)
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    cost = path_cost(path, mapping)
    assert cost == 20


def test_part1_steps():
    data = [
        "2413432311323",
        "3215453535623",
        "3255245654254",
        "3446585845452",
        "4546657867536",
        "1438598798454",
        "4457876987766",
        "3637877979653",
        "4654967986887",
        "4564679986453",
        "1224686865563",
        "2546548887735",
        "4322674655533",
    ]
    mapping = parser(data)
    print_grid_and_path(mapping, path=[])
    cost = dijkstra(mapping, (0, 0), (12, 12))
    logger.debug(cost)
    assert cost == 102


def test_part2_steps():
    data = [
        "2413432311323",
        "3215453535623",
        "3255245654254",
        "3446585845452",
        "4546657867536",
        "1438598798454",
        "4457876987766",
        "3637877979653",
        "4654967986887",
        "4564679986453",
        "1224686865563",
        "2546548887735",
        "4322674655533",
    ]
    mapping = parser(data)
    print_grid_and_path(mapping, path=[])
    cost = dijkstra(mapping, (0, 0), (12, 12), min_step=4, step_limit=10)
    logger.debug(cost)
    assert cost == 94


def test_part1():
    data = [
        "2413432311323",
        "3215453535623",
        "3255245654254",
        "3446585845452",
        "4546657867536",
        "1438598798454",
        "4457876987766",
        "3637877979653",
        "4654967986887",
        "4564679986453",
        "1224686865563",
        "2546548887735",
        "4322674655533",
    ]
    cost = part1(data)
    assert cost == 102


def test_part2():
    data = [
        "2413432311323",
        "3215453535623",
        "3255245654254",
        "3446585845452",
        "4546657867536",
        "1438598798454",
        "4457876987766",
        "3637877979653",
        "4654967986887",
        "4564679986453",
        "1224686865563",
        "2546548887735",
        "4322674655533",
    ]
    assert part2(data) == 94
