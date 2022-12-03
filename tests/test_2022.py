import pytest
from utils import load_file
import jax.numpy as jnp

from year_2022 import (day_1, day_2, day_3)


def test_grouping():
    raw_list = ["1000","2000","3000","","4000","",
                "5000","6000","","7000","8000","9000",
                "","10000"] 
    formatted = day_1.grouping(raw_list)
    assert len(formatted) == 5
    assert isinstance(formatted[0], jnp.ndarray)

@pytest.mark.parametrize("input,expected", [(["1000","2000","3000","","4000","",
                                             "5000","6000","","7000","8000","9000",
                                             "","10000"], 24000)])
def test_day_1_1(input, expected):
    input = day_1.grouping(input)
    result = day_1.part_1(input)
    assert result == expected


@pytest.mark.parametrize("input,expected", [(["1000","2000","3000","","4000","",
                                             "5000","6000","","7000","8000","9000",
                                             "","10000"], 45000)])
def test_day_1_2(input, expected):
    input = day_1.grouping(input)
    result = day_1.part_2(input)
    assert result == expected


def test_day_2_1():
    inputs = ["A Y", "B X", "C Z"]
    result = day_2.part_1(inputs)
    assert result == 15

def test_day_2_2():
    inputs = ["A Y", "B X", "C Z"]
    result = day_2.part_2(inputs)
    assert result == 12


@pytest.mark.parametrize("inputs,expected", [("A Y", ("rock", "paper")),
                                             ("B X", ("paper", "rock")),
                                             ("C Z", ("sissors", "sissors"))])
def test_game_inputs(inputs, expected):
    code = dict(opponent=dict(A="rock", B="paper", C="sissors"),
            player=dict(Y="paper", X="rock", Z="sissors"))
    opp, player = day_2.game_inputs(inputs, code)
    assert opp == expected[0]
    assert player == expected[1]


@pytest.mark.parametrize("inputs,expected", [(("rock", "paper"), 1),
                                             (("paper", "rock"), -1),
                                             (("rock", "rock"), 0),
                                             (("sissors", "paper"), -1),
                                             (("rock", "sissors"), -1)])    
def test_rock_paper_sissors_game_outcome(inputs, expected):
    result = day_2.game_outcome(inputs[0], inputs[1])
    assert result == expected

def test_game_score():
    assert day_2.score("rock", -1) == 1
    assert day_2.score("paper", 1) == 8
    assert day_2.score("sissors", 0) == 6


@pytest.mark.parametrize("inputs,expected", [(("rock", 0), "rock"),
                                             (("paper", -1), "rock"),
                                             (("sissors", 1), "rock")
                                             ])    
def test_rock_paper_sissors_choice_from_outcome(inputs, expected):
    result = day_2.choice_from_outcome(inputs[0], inputs[1])
    assert result == expected


@pytest.mark.parametrize("inputs,expected", [("asdjhasd", 4), ("aaaa", 2), ])
def test_split_lists_in_half(inputs, expected):
    assert len(day_3.split_lists_in_half(inputs)[0]) == expected
    
@pytest.mark.parametrize("inputs,expected", [("vJrwpWtwJgWrhcsFMMfFFhFp", "p"),
                                             ("jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL", "L"),
                                             ("wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn", "v"),
                                             ("PmmdzqPrVvPwwTWBwg", "P"),
                                             ("ttgJtRGJQctTZtZT", "t"),
                                             ("CrZsJsPPZsGzwwsLwLmpwMDw", "s"),
                                             ])
def test_find_double_char(inputs, expected):
    comp_1, comp_2 = day_3.split_lists_in_half(inputs)
    assert day_3.find_double_char(comp_1, comp_2) == expected

def test_score_char():
    assert day_3.score_char("a") == 1
    assert day_3.score_char("b") == 2
    assert day_3.score_char("z") == 26
    assert day_3.score_char("Z") == 52

@pytest.mark.parametrize("inputs,expected", [(["vJrwpWtwJgWrhcsFMMfFFhFp",
                                             "jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL",
                                             "wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn",
                                             "PmmdzqPrVvPwwTWBwg",
                                             "ttgJtRGJQctTZtZT",
                                             "CrZsJsPPZsGzwwsLwLmpwMDw",
                                             ], 157)])
def test_day3_1(inputs, expected):
    result = day_3.part_1(inputs)
    assert result == expected


@pytest.mark.parametrize("inputs,expected", [(["vJrwpWtwJgWrhcsFMMfFFhFp",
                                               "jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL",
                                                "PmmdzqPrVvPwwTWBwg",
                                             "wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn",
                                             "ttgJtRGJQctTZtZT",
                                             "CrZsJsPPZsGzwwsLwLmpwMDw"
                                             ], 70)])
def test_day3_1(inputs, expected):
    result = day_3.part_2(inputs)
    assert result == expected