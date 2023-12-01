import pytest
from year_2017 import day_7, day_8, day_9

from utils import load_file


@pytest.mark.parametrize(
    "input,expected", [("tests/test_data/data_2017_7_1.txt", "tknk")]
)
def test_day_7_1(input, expected):
    input = load_file(input)
    input = day_7.parse_input_to_graph(input)
    result = day_7.part_1(input)
    assert result == expected


@pytest.mark.parametrize("input,expected", [("tests/test_data/data_2017_7_1.txt", 8)])
def test_day_7_2(input, expected):
    input = load_file(input)
    input = day_7.parse_input_to_graph(input)
    base_node = "tknk"
    result = day_7.part_2(input, base_node)
    assert result[0] == expected
    assert result[1] == [251, 243, 243]


@pytest.mark.parametrize(
    "input",
    [
        (["A (1)", "B (2) -> C, D", "C (3)", "D (4) -> E"]),
        (["C (3)", "D (4) -> E", "A (1)", "B (2) -> C, D"]),
    ],
)
def test_input_to_graph(input):
    result = day_7.parse_input_to_graph(input)
    assert result["A"]["value"] == 1
    assert result["B"]["value"] == 2
    assert result["B"]["out"] == ["C", "D"]
    assert result["C"]["value"] == 3
    assert result["C"]["rec"] == ["B"]
    assert result["D"]["value"] == 4
    assert result["D"]["rec"] == ["B"]
    assert result["D"]["out"] == ["E"]


def test_text_to_graph():
    input = load_file("tests/test_data/data_2017_7_1.txt")
    result = day_7.parse_input_to_graph(input)
    assert result["ugml"]["rec"] == ["tknk"]
    assert result["ugml"]["out"] == ["gyxo", "ebii", "jptl"]
    assert result["ebii"]["rec"] == ["ugml"]


def test_day_8_1():
    inputs = load_file("tests/test_data/data_2017_8.txt")
    register = day_8.get_register(inputs)
    result = day_8.part_1(inputs, register)
    assert result == 1


def test_day_8_2():
    inputs = load_file("tests/test_data/data_2017_8.txt")
    register = day_8.get_register(inputs)
    result = day_8.part_2(inputs, register)
    assert result == 10


def test_get_register():
    inputs = load_file("tests/test_data/data_2017_8.txt")
    register = day_8.get_register(inputs)
    assert sorted(list(register.keys())) == ["a", "b", "c"]
    assert register["a"] == 0


@pytest.mark.parametrize(
    "input,expected",
    [
        ("bac inc 5 if a > 1", {"reg": "bac", "inst": 5, "cond": "a > 1"}),
        ("c dec -10 if a >= 1", {"reg": "c", "inst": 10, "cond": "a >= 1"}),
        ("c inc -20 if c == 10", {"reg": "c", "inst": -20, "cond": "c == 10"}),
    ],
)
def test_parse_line(input, expected):
    parsed = day_8.parse_line(input)
    assert parsed == expected


@pytest.mark.parametrize(
    "input,expected", [("a == 0", True), ("b < 10", True), ("c != 10", False)]
)
def test_eval_cond(input, expected):
    reg = dict(a=0, b=5, c=10)
    result = day_8.eval_cond(reg, input)
    assert result == expected


@pytest.mark.parametrize(
    "inputs,expected",
    [
        ("{}", 1),
        ("{{{}}}", 3),
        ("{{},{}}", 3),
        ("{{{},{},{{}}}}", 6),
        ("{<{},{},{{}}>}", 1),
        ("{<a>,<a>,<a>,<a>}", 1),
        ("{{<a>},{<a>},{<a>},{<a>}}", 5),
        ("{{<!>},{<!>},{<!>},{<a>}}", 2),
    ],
)
def test_get_groups(inputs, expected):
    score, groups, _ = day_9.get_groups(inputs)
    assert groups == expected


@pytest.mark.parametrize(
    "inputs,expected",
    [
        ("{}", 1),
        ("{{{}}}", 6),
        ("{{},{}}", 5),
        ("{{{},{},{{}}}}", 16),
        ("{<a>,<a>,<a>,<a>}", 1),
        ("{{<ab>},{<ab>},{<ab>},{<ab>}}", 9),
        ("{{<!!>},{<!!>},{<!!>},{<!!>}}", 9),
        ("{{<a!>},{<a!>},{<a!>},{<ab>}}", 3),
    ],
)
def test_get_score(inputs, expected):
    score, groups, _ = day_9.get_groups(inputs)
    assert score == expected


@pytest.mark.parametrize(
    "inputs,expected",
    [
        ("<>", 0),
        ("<random characters>", 17),
        ("<<<<>", 3),
        ("<{!>}>", 2),
        ("<!!>", 0),
        ("<!!!>>", 0),
        ('<{oi"!a,<{i<a>', 10),
    ],
)
def test_get_garb(inputs, expected):
    _, _, garb_counter = day_9.get_groups(inputs)
    assert garb_counter == expected
