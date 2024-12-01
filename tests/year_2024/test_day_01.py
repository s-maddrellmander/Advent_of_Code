from solutions.year_2024.day_01 import *
import pytest

@pytest.fixture
def input_data():
    data = ["3   4",
           "4   3",
           "2   5",
           "1   3",
           "3   9",
           "3   3",]
    return data 


def test_input_parse(input_data):
    parsed = parse_input(input_data)
    assert parsed[0] == (3, 4)
    assert len(parsed) == 6

def test_part1(input_data):
    res = part1(input_data)
    assert res == 11

def test_part2(input_data):
    res = part2(input_data)
    assert res == 31
