import pytest

from solutions.year_2023.day_08 import *


def test_parse_input():
    raw_input = [
        "RL",
        "",
        "AAA = (BBB, CCC)",
        "BBB = (DDD, EEE)",
        "CCC = (ZZZ, GGG)",
        "DDD = (DDD, DDD)",
        "EEE = (EEE, EEE)",
        "GGG = (GGG, GGG)",
        "ZZZ = (ZZZ, ZZZ)",
    ]
    graph, instructions = parse_input(raw_input)
    assert instructions == "RL"
    assert graph == {
        "AAA": {"left": "BBB", "right": "CCC"},
        "BBB": {"left": "DDD", "right": "EEE"},
        "CCC": {"left": "ZZZ", "right": "GGG"},
        "DDD": {"left": "DDD", "right": "DDD"},
        "EEE": {"left": "EEE", "right": "EEE"},
        "GGG": {"left": "GGG", "right": "GGG"},
        "ZZZ": {"left": "ZZZ", "right": "ZZZ"},
    }


def test_traverse_graph():
    raw_input = [
        "RL",
        "",
        "AAA = (BBB, CCC)",
        "BBB = (DDD, EEE)",
        "CCC = (ZZZ, GGG)",
        "DDD = (DDD, DDD)",
        "EEE = (EEE, EEE)",
        "GGG = (GGG, GGG)",
        "ZZZ = (ZZZ, ZZZ)",
    ]
    graph, instructions = parse_input(raw_input)
    path = traverse_graph(graph, instructions)
    assert path == ["CCC", "ZZZ"]


def test_part1():
    raw_input = [
        "RL",
        "",
        "AAA = (BBB, CCC)",
        "BBB = (DDD, EEE)",
        "CCC = (ZZZ, GGG)",
        "DDD = (DDD, DDD)",
        "EEE = (EEE, EEE)",
        "GGG = (GGG, GGG)",
        "ZZZ = (ZZZ, ZZZ)",
    ]
    assert part1(raw_input) == 2


def test_multiple_start_and_end():
    raw_input = [
        "LR",
        "",
        "11A = (11B, XXX)",
        "11B = (XXX, 11Z)",
        "11Z = (11B, XXX)",
        "22A = (22B, XXX)",
        "22B = (22C, 22C)",
        "22C = (22Z, 22Z)",
        "22Z = (22B, 22B)",
        "XXX = (XXX, XXX)",
    ]
    graph, instructions = parse_input(raw_input)
    starts, ends = get_all_start_and_end_nodes(graph)
    assert starts == ["11A", "22A"]
    assert ends == ["11Z", "22Z"]


def test_part2():
    raw_input = [
        "LR",
        "",
        "11A = (11B, XXX)",
        "11B = (XXX, 11Z)",
        "11Z = (11B, XXX)",
        "22A = (22B, XXX)",
        "22B = (22C, 22C)",
        "22C = (22Z, 22Z)",
        "22Z = (22B, 22B)",
        "XXX = (XXX, XXX)",
    ]
    assert part2(raw_input) == 6
