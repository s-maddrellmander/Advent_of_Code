import pytest

from solutions.year_2023.day_04 import *


def test_calculate_score():
    assert calculate_score({"A", "B", "C"}) == 4
    assert calculate_score({"A", "B"}) == 2
    assert calculate_score({"A"}) == 1
    assert calculate_score(set()) == 0


def test_parse_cards_to_dictionaries():
    data = ["Card 1: 2 3 4 | 5 6 7", "Card 2: 8 9 10 | 11 12 13"]
    result = parse_cards_to_dictionaries(data)
    expected = {
        1: [set([2, 3, 4]), set([5, 6, 7])],
        2: [set([8, 9, 10]), set([11, 12, 13])],
    }
    assert result == expected

    data = ["Card 3: 14 15 16 | 17 18 19", "Card 4: 20 21 22 | 23 24 25"]
    result = parse_cards_to_dictionaries(data)
    expected = {
        3: [set([14, 15, 16]), set([17, 18, 19])],
        4: [set([20, 21, 22]), set([23, 24, 25])],
    }
    assert result == expected


def test_part1():
    cards = [
        "Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53",
        "Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19",
        "Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1",
        "Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83",
        "Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36",
        "Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11",
    ]
    assert part1(cards) == 13


def test_part2():
    cards = [
        "Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53",
        "Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19",
        "Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1",
        "Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83",
        "Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36",
        "Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11",
    ]
    assert part2(cards) == 30


def test_debug_part2():
    cards = [
        "Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53",
        "Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19",
        "Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1",
        "Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83",
        "Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36",
        "Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11",
    ]
    parsed_cards = parse_cards(cards)
    output = process_cards(parsed_cards)
    assert {1: 1, 2: 2, 3: 4, 4: 8, 5: 14, 6: 1} == output
    assert sum(process_cards(parsed_cards).values()) == 30
