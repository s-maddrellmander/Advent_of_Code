import pytest

from solutions.year_2024.day_11 import *


def ret_val(idx):
    if idx % 2 == 0:
        return [idx]
    else:
        return [idx, idx]


def test_ret_val():
    example = [x for i in range(10) for x in ret_val(i)]
    assert example == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9]


def test_parse_input():
    input_string = ["1 2 3 4 5 6 7 8 9 10"]
    expected = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    assert parse_input(input_string) == expected

    input_string = ["0 1 10 99 999"]
    expected = ["0", "1", "10", "99", "999"]
    assert parse_input(input_string) == expected


def test_blink():
    stones = ["0", "1", "10", "99", "999"]
    expected = [str(x) for x in [1, 2024, 1, 0, 9, 9, 2021976]]
    assert blink(stones) == expected


def test_fast_blink():
    stones = ["0", "1", "10", "99", "999"]
    stones = {stone: 1 for stone in stones}
    expected = {"1": 2, "0": 1, "9": 2, "2021976": 1, "2024": 1}
    assert fast_blink(stones) == expected


def test_6x_blinks():
    stones = ["125", "17"]

    stones = blink(stones)
    # After 1 blink:
    expeted = ["253000", "1", "7"]
    assert len(stones) == len(expeted)
    assert stones == expeted

    # After 2 blinks:
    expeted = ["253", "0", "2024", "14168"]
    stones = blink(stones)
    assert len(stones) == len(expeted)
    assert stones == expeted

    # After 3 blinks:
    expeted = ["512072", "1", "20", "24", "28676032"]
    stones = blink(stones)
    assert len(stones) == len(expeted)
    assert stones == expeted

    # After 4 blinks:
    expeted = ["512", "72", "2024", "2", "0", "2", "4", "2867", "6032"]
    stones = blink(stones)
    assert len(stones) == len(expeted)
    assert stones == expeted


def test_part1():
    input_data = ["125 17"]
    expected = 55312
    assert part1(input_data) == expected
