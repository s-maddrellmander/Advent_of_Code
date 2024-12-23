import pytest

from solutions.year_2024.day_08 import *


@pytest.fixture
def data():
    return [
        "............",
        "........0...",
        ".....0......",
        ".......0....",
        "....0.......",
        "......A.....",
        "............",
        "............",
        "........A...",
        ".........A..",
        "............",
        "............",
    ]


def test_parse_data(data):
    nodes = parse_data(data)
    assert len(nodes["0"]) == 4
    assert len(nodes["A"]) == 3

    assert nodes["0"][0] == complex(8, 1)


def test_all_pairs(data):
    nodes = parse_data(data)
    pairs = find_all_pairs(nodes["0"])
    assert len(pairs) == 6
    assert pairs[0] == (complex(8, 1), complex(5, 2))


def test_complex_maths():
    a = 1 + 1j
    b = 2 + 2j  # Therefore the distance should be 1 + 1j
    assert (b - a) == 1 + 1j

    delta = b - a

    # Know that a +/- delta = b/antinode - but we don't know which
    for sign in [-1, 1]:
        if a + sign * delta == b:
            continue
        else:
            assert a + sign * delta != b


@pytest.mark.parametrize(
    "pair, expected",
    [((1 + 1j, 2 + 2j), (0 + 0j, 3 + 3j)), ((2 + 2j, 3 + 1j), (1 + 3j, 4 + 0j))],
)
def test_return_antinode(pair, expected):
    assert return_antinode(*pair) == expected[0]
    assert return_antinode(*pair[::-1]) == expected[1]


def test_part1(data):
    assert part1(data) == 14


def test_part2(data):
    assert part2(data) == 34


def test_part2_simple():
    data = [
        "T.........",
        "...T......",
        ".T........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
    ]
    assert part2(data) == 9
