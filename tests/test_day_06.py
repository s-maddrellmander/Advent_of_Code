import pytest

from solutions.year_2023.day_06 import *


def test_quadratic_solution():
    # test a range of values
    assert quadratic_solution(7, 9) == (2, 5)
    assert quadratic_solution(15, 40) == (4, 11)
    assert quadratic_solution(30, 200) == (11, 19)


def test_part1():
    input_data = ["Time:      7  15   30", "Distance:  9  40  200"]
    assert part1(input_data) == 288


def test_part2():
    input_data = ["Time:      7  15   30", "Distance:  9  40  200"]
    assert part2(input_data) == 71503
