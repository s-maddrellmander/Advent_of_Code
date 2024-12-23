import pytest

from solutions.year_2024.day_10 import *


@pytest.fixture
def simple_data():
    return [
        "...0...",
        "...1...",
        "...2...",
        "6543456",
        "7.....7",
        "8.....8",
        "9.....9",
    ]


@pytest.fixture
def data():
    return [
        "89010123",
        "78121874",
        "87430965",
        "96549874",
        "45678903",
        "32019012",
        "01329801",
        "10456732",
    ]


@pytest.fixture
def mid_data():
    return [
        "..90..9",
        "...1.98",
        "...2..7",
        "6543456",
        "765.987",
        "876....",
        "987....",
    ]


def test_get_map(simple_data):
    top_map, trail_heads = get_map(simple_data)
    assert len(top_map) == 16

    assert top_map[3 + 0j] == 0
    assert top_map[3 + 1j] == 1
    assert top_map[0 + 6j] == 9
    assert trail_heads == [3 + 0j]


def test_bfs(simple_data):
    top_map, trail_heads = get_map(simple_data)

    num_trails, trails_end = bfs(top_map, trail_heads[0])
    assert num_trails == 2
    assert set(trails_end) == set([0 + 6j, 6 + 6j])


def test_part1(data, mid_data):
    assert part1(mid_data) == 4
    assert part1(data) == 36


def test_part2(data, mid_data):
    assert part2(mid_data) == 13
    assert part2(data) == 81
