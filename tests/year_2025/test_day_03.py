from solutions.year_2025.day_03 import *
import pytest

test_data = [
    "987654321111111",
    "811111111111119",
    "234234234234278",
    "818181911112111"
]

def test_part1():
    result = part1(test_data)
    assert result == 357