import pytest

from solutions.year_2024.day_07 import *


@pytest.fixture
def data():
    data = [
        "190: 10 19",
        "3267: 81 40 27",
        "83: 17 5",
        "156: 15 6",
        "7290: 6 8 6 15",
        "161011: 16 10 13",
        "192: 17 8 14",
        "21037: 9 7 18 13",
        "292: 11 6 16 20",
    ]
    return data


def test_parse_input(data):
    parsed_lines = parse_lines(data)

    assert parsed_lines[0] == (190, [10, 19])
    assert len(parsed_lines) == 9

    eg = ["17406915: 29 6 1 2 4 6 484 4 9 1 9 6"]
    parsed_eg = parse_lines(eg)
    assert parsed_eg[0][0] == 17406915
    assert len(parsed_eg[0][1]) == 12

    line = ["19384474050: 6 3 8 2 53 3 447 40 47 6"]
    res = part1(line)
    assert res == 0


def test_problem():
    line = ["1194: 74 824 72 1 223 3"]
    res = part1(line)
    assert res == 0


@pytest.mark.parametrize(
    "current,operator,new_value,expected", [(2, "+", 3, 5), (2, "*", 3, 6)]
)
def test_do_maths(current, operator, new_value, expected):
    assert do_maths(current, operator, new_value) == expected


def test_branching_op():
    values = [10, 19]
    target = 190

    ret = branching_operation(values, target)


def test_part1(data):
    res = part1(data)
    assert res == 3749


def test_long():
    values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    target = 1024
    assert branching_operation(values, target)

    target = 292548726
    values = [2, 4, 7, 4, 4, 2, 9, 9, 7, 5, 38, 27]
    assert branching_operation(values, target) == False

    target = 1610070872006
    values = [88, 8, 56, 906, 3, 73, 25, 8, 6]

    assert branching_operation(values, target) == False


def test_part1(data):
    assert part2(data) == 11387
