import pytest

from solutions.year_2023.day_20 import *

SIMPLE_DATA = [
    "broadcaster -> a, b, c",
    "%a -> b",
    "%b -> c",
    "%c -> inv",
    "&inv -> a",
]

COMPLEX_DATA = [
    "broadcaster -> a",
    "%a -> inv, con",
    "&inv -> b",
    "%b -> con",
    "&con -> output",
]


def test_parse():
    nodes = parse_input_to_graph(SIMPLE_DATA)
    assert nodes["broadcaster"].out_edges == {"a", "b", "c"}
    assert nodes["a"].out_edges == {"b"}
    assert nodes["b"].out_edges == {"c"}
    assert nodes["c"].out_edges == {"inv"}
    assert nodes["inv"].out_edges == {"a"}

    # Test in edges
    assert nodes["broadcaster"].in_edges == set()
    assert nodes["a"].in_edges == {"broadcaster", "inv"}
    assert nodes["b"].in_edges == {"broadcaster", "a"}
    assert nodes["c"].in_edges == {"broadcaster", "b"}
    assert nodes["inv"].in_edges == {"c"}

    # draw_graph(nodes)


def test_propagate_value():
    nodes = parse_input_to_graph(SIMPLE_DATA)
    flip_fop_values = propagate_value(nodes, "broadcaster", 0)
    assert flip_fop_values == {"a": False, "b": False, "c": False}
    low, high, details = count_counts(nodes=nodes)
    assert details == {
        "broadcaster": (1, 0),
        "a": (2, 1),
        "b": (2, 1),
        "c": (2, 1),
        "inv": (1, 1),
    }
    assert low == 8
    assert high == 4
    flip_fop_values = propagate_value(nodes, "broadcaster", 0)
    assert flip_fop_values == {"a": False, "b": False, "c": False}


def test_easy_example_1000():
    nodes = parse_input_to_graph(SIMPLE_DATA)
    for button_press in range(1000):
        propagate_value(nodes, "broadcaster", 0)
    low, high, details = count_counts(nodes=nodes)
    assert low == 8000
    assert high == 4000
    assert low * high == 32000000


def test_complex():
    nodes = parse_input_to_graph(COMPLEX_DATA)
    propagate_value(nodes, "broadcaster", 0)
    low, high, details = count_counts(nodes=nodes)
    assert low == high == 4
    propagate_value(nodes, "broadcaster", 0)
    low, high, details = count_counts(nodes=nodes)
    assert low == 8
    assert high == 6
    propagate_value(nodes, "broadcaster", 0)
    low, high, details = count_counts(nodes=nodes)
    assert low == 13
    assert high == 9


def test_complex_1000():
    nodes = parse_input_to_graph(COMPLEX_DATA)
    for button_press in range(1000):
        propagate_value(nodes, "broadcaster", 0)
    low, high, details = count_counts(nodes=nodes)
    assert low == 4250
    assert high == 2750
    assert low * high == 11687500


@pytest.mark.parametrize(
    "data, val", [(SIMPLE_DATA, 32000000), (COMPLEX_DATA, 11687500)]
)
def test_part1(data, val):
    assert part1(data) == val
