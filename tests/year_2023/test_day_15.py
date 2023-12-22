import pytest

from solutions.year_2023.day_15 import *


def test_hash_function():
    current = 0
    current = hash_function("H", current_value=current)
    assert current == 200
    current = hash_function("A", current_value=current)
    assert current == 153
    current = hash_function("S", current_value=current)
    assert current == 172
    current = hash_function("H", current_value=current)
    assert current == 52


def test_parse_input():
    text = "rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7"
    inputs = parse_input(text)
    assert len(inputs) == 11


def test_part1():
    text = ["rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7\n"]
    assert part1(input_data=text) == 1320


def test_adding_LL():
    boxes = {0: None, 1: None, 3: None}
    boxes[0] = add_to_linked_list(boxes[0], name="rn", value=1)
    assert boxes[0].name == "rn"
    assert boxes[0].value == 1
    assert boxes[0].next is None
    boxes[1] = add_to_linked_list(boxes[1], name="qp", value=3)
    assert boxes[1].name == "qp"
    boxes[0] = add_to_linked_list(boxes[0], name="cm", value=2)
    assert boxes[0].name == "rn"
    assert boxes[0].value == 1
    assert boxes[0].next.name == "cm"
    assert boxes[0].next.value == 2


def test_removing_LL():
    boxes = {0: None, 1: None, 3: None}
    boxes[0] = add_to_linked_list(boxes[0], name="rn", value=1)
    boxes[1] = add_to_linked_list(boxes[1], name="qp", value=3)
    boxes[0] = add_to_linked_list(boxes[0], name="cm", value=2)

    boxes[0] = remove_from_linked_list(boxes[0], name="rn")
    assert boxes[0].name == "cm"


def test_part2():
    text = ["rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7\n"]
    assert part2(input_data=text) == 145
