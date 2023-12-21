import pytest

from solutions.year_2023.day_19 import *

INPUT_DATA = [
    "px{a<2006:qkq,m>2090:A,rfg}",
    "pv{a>1716:R,A}",
    "lnx{m>1548:A,A}",
    "rfg{s<537:gd,x>2440:R,A}",
    "qs{s>3448:A,lnx}",
    "qkq{x<1416:A,crn}",
    "crn{x>2662:A,R}",
    "in{s<1351:px,qqz}",
    "qqz{s>2770:qs,m<1801:hdj,R}",
    "gd{a>3333:R,R}",
    "hdj{m>838:A,pv}",
    "",
    "{x=787,m=2655,a=1222,s=2876}",
    "{x=1679,m=44,a=2067,s=496}",
    "{x=2036,m=264,a=79,s=2244}",
    "{x=2461,m=1339,a=466,s=291}",
    "{x=2127,m=1623,a=2188,s=1013}",
]


def test_parsing():
    nodes, values = parse_input(INPUT_DATA)
    assert sorted(list(nodes.keys())) == sorted(
        ["px", "pv", "lnx", "rfg", "qs", "qkq", "crn", "in", "qqz", "gd", "hdj"]
    )
    assert values == [
        {"x": 787, "m": 2655, "a": 1222, "s": 2876},
        {"x": 1679, "m": 44, "a": 2067, "s": 496},
        {"x": 2036, "m": 264, "a": 79, "s": 2244},
        {"x": 2461, "m": 1339, "a": 466, "s": 291},
        {"x": 2127, "m": 1623, "a": 2188, "s": 1013},
    ]

    assert nodes["in"] == [["s<1351", "px"], ["qqz"]]


def test_calc_value():
    nodes, values = parse_input(INPUT_DATA)
    assert calculate_value(values[0], nodes) == "A"
    assert calculate_value(values[1], nodes) == "R"
    assert calculate_value(values[2], nodes) == "A"
    assert calculate_value(values[3], nodes) == "R"
    assert calculate_value(values[4], nodes) == "A"


def test_part1():
    assert part1(INPUT_DATA) == 19114


@pytest.mark.skip(reason="Not yet implemented")
def test_part2():
    assert part2(INPUT_DATA) == 167409079868000
