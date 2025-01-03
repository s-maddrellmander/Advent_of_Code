import pytest

from solutions.year_2024.day_05 import *


def test_parse_rule():
    rules = ["0|1", "1|2", "2|3", "2|4"]

    assert parse_rule(rules)[0] == {0: [1], 1: [2], 2: [3, 4]}


def test_parse_pages():
    pages = ["1,2,3", "4,5,6", "7,8,9"]

    assert parse_pages(pages) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_check_page():
    rules = [
        "47|53",
        "97|13",
        "97|61",
        "97|47",
        "75|29",
        "61|13",
        "75|53",
        "29|13",
        "97|29",
        "53|29",
        "61|53",
        "97|53",
        "61|29",
        "47|13",
        "75|47",
        "97|75",
        "47|61",
        "75|61",
        "47|29",
        "75|13",
        "53|13",
    ]
    rules = parse_rule(rules)[0]

    page = [75, 47, 61, 53, 29]

    assert check_page(page, rules) == True


pages = [
    [75, 47, 61, 53, 29],
    [97, 61, 53, 29, 13],
    [75, 29, 13],
    [75, 97, 47, 61, 53],
    [61, 13, 29],
    [97, 13, 75, 29, 47],
]
flags = [True, True, True, False, False, False]


@pytest.mark.parametrize("page, flag", zip(pages, flags))
def test_check_all_pages(page, flag):
    rules = [
        "47|53",
        "97|13",
        "97|61",
        "97|47",
        "75|29",
        "61|13",
        "75|53",
        "29|13",
        "97|29",
        "53|29",
        "61|53",
        "97|53",
        "61|29",
        "47|13",
        "75|47",
        "97|75",
        "47|61",
        "75|61",
        "47|29",
        "75|13",
        "53|13",
    ]
    rules = parse_rule(rules)[0]

    assert (
        check_page(page, rules) == flags[pages.index(page)]
    ), f"Failed for page {page}"


def test_part1():
    input_data = [
        "47|53",
        "97|13",
        "97|61",
        "97|47",
        "75|29",
        "61|13",
        "75|53",
        "29|13",
        "97|29",
        "53|29",
        "61|53",
        "97|53",
        "61|29",
        "47|13",
        "75|47",
        "97|75",
        "47|61",
        "75|61",
        "47|29",
        "75|13",
        "53|13",
        "",
        "75,47,61,53,29",
        "97,61,53,29,13",
        "75,29,13",
        "75,97,47,61,53",
        "61,13,29",
        "97,13,75,29,47",
    ]

    assert part1(input_data) == 143


@pytest.mark.parametrize(
    "page, expected", [([75, 97, 47, 61, 53], [97, 75, 47, 61, 53])]
)
def test_fix_page(page, expected):
    rules = [
        "47|53",
        "97|13",
        "97|61",
        "97|47",
        "75|29",
        "61|13",
        "75|53",
        "29|13",
        "97|29",
        "53|29",
        "61|53",
        "97|53",
        "61|29",
        "47|13",
        "75|47",
        "97|75",
        "47|61",
        "75|61",
        "47|29",
        "75|13",
        "53|13",
    ]
    rules, reverse_rules = parse_rule(rules)

    assert fix_page(page, rules, reverse_rules) == expected


def test_part2():
    input_data = [
        "47|53",
        "97|13",
        "97|61",
        "97|47",
        "75|29",
        "61|13",
        "75|53",
        "29|13",
        "97|29",
        "53|29",
        "61|53",
        "97|53",
        "61|29",
        "47|13",
        "75|47",
        "97|75",
        "47|61",
        "75|61",
        "47|29",
        "75|13",
        "53|13",
        "",
        "75,47,61,53,29",
        "97,61,53,29,13",
        "75,29,13",
        "75,97,47,61,53",
        "61,13,29",
        "97,13,75,29,47",
    ]

    assert part2(input_data) == 123
