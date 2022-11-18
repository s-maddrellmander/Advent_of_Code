import pytest
from utils import load_file

from year_2017 import (day_7)

@pytest.mark.parametrize("input,expected", [("tests/test_data/data_2017_7_1.txt", "tknk")])
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

@pytest.mark.parametrize("input", [(["A (1)", "B (2) -> C, D", "C (3)", "D (4) -> E"]),
                                   (["C (3)", "D (4) -> E", "A (1)", "B (2) -> C, D"])])
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


