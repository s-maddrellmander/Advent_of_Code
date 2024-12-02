import pytest

from solutions.year_2024.day_02 import *


@pytest.fixture
def input_data():
    test_data = [
        "7 6 4 2 1",
        "1 2 7 8 9",
        "9 7 6 2 1",
        "1 3 2 4 5",
        "8 6 4 4 1",
        "1 3 6 7 9",
    ]
    return test_data


def test_parse(input_data):
    lines = parse_lines(input_data)
    assert lines[0] == [7, 6, 4, 2, 1]
    assert len(lines) == 6


def test_is_safe():
    assert is_safe([7, 6, 4, 2, 1]) == True
    assert is_safe([1, 3, 6, 7, 9]) == True
    assert is_safe([1, 2, 7, 8, 9]) == False


def test_part_1(input_data):
    res = part1(input_data)
    assert res == 2


def test_part_2(input_data):
    res = part2(input_data)
    assert res == 4


def test_make_combs(input_data):
    lines = parse_lines(input_data)
    new_lines = make_combs(lines[0])

    assert len(new_lines) == 5
