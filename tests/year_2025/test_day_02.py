from solutions.year_2025.day_02 import *
import pytest

test_data = [
    "11-22","95-115","998-1012","1188511880-1188511890","222220-222224",
"1698522-1698528","446443-446449","38593856-38593862","565653-565659",
"824824821-824824827","2121212118-2121212124"
]

def test_part1():
    assert part1(test_data) == 1227775554


def test_part2():
    assert part2(test_data) == 4174379265 

@pytest.mark.parametrize("ran", [(11, 22, 2), (95, 115, 1), (1188511880,1188511890, 1)])
def test_scan_range(ran):
    q, v  = scan_range(ran[0], ran[1]+1)
    assert q == ran[2]


@pytest.mark.parametrize("ran", [(11, 22, 2), (95, 115, 2), (1188511880,1188511890, 1), (446443,446449, 1)])
def test_scan_rangei_full(ran):
    q, v  = scan_range_full(ran[0], ran[1]+1)
    print(q, v)
    assert q == ran[2]




