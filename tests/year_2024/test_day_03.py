from solutions.year_2024.day_03 import *
import pytest


@pytest.fixture
def data():
    return ["xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5)))"]



def test_match_pattern(data):
    res = match_pattern(data, "mul")



def test_part1(data):
    assert part1(data) == 161

def test_part2():
    data2 = ["xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"]
    assert part2(data2) == 48
