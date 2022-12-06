import pytest
from utils import load_file, Queue, Cache
import jax.numpy as jnp

from year_2022 import (day_1, day_2, day_3, day_4, day_5, day_6)


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
    

def test_get_ranges():
    inputs = "2-4,6-8"
    r1, r2 = day_4.get_ranges(inputs)
    assert jnp.allclose(r1, jnp.arange(2, 5, 1))
    assert jnp.allclose(r2, jnp.arange(6, 9, 1))
    assert jnp.max(r1) == 4


@pytest.mark.parametrize("inputs,expected", [((0, 3, 4, 6), 0),
                                             ((0, 9, 4, 6), 1),
                                             ((4, 6, 3, 6), 1),
                                             ((4, 6, 6, 9), 0),
                         ])
def test_check_intersection(inputs, expected):
    r1 = jnp.arange(inputs[0], inputs[1])    
    r2 = jnp.arange(inputs[2], inputs[3])    
    assert day_4.check_intersection(r1, r2) == expected

def test_day_4_1():
    inputs = ["2-4,6-8",
                "2-3,4-5",
                "5-7,7-9",
                "2-8,3-7",
                "6-6,4-6",
                "2-6,4-8"]
    assert day_4.part_1(inputs) == 2

def test_day_4_2():
    inputs = ["2-4,6-8",
                "2-3,4-5",
                "5-7,7-9",
                "2-8,3-7",
                "6-6,4-6",
                "2-6,4-8"]
    assert day_4.part_2(inputs) == 4
    
def test_build_stacks():
    stacks = {1: "ZN", 2: "MCD", 3: "P"}
    stacks = day_5.get_all_stacks(stacks)
    assert len(stacks[1]) == 2
    assert len(stacks[2]) == 3
    assert len(stacks[3]) == 1
    
    x = stacks[1].pop()
    assert x == "N"
    stacks[3].append(x)
    assert len(stacks[3]) == 2
    assert stacks[3].pop() == "N"
    assert stacks[3].pop() == "P"

def test_day_5_1():
    stacks = {1: "ZN", 2: "MCD", 3: "P"}
    stacks = day_5.get_all_stacks(stacks)
    inputs = load_file('tests/test_data/data_2022_5.txt')
    results = day_5.part_1(stacks, inputs)
    assert results == "CMZ"

def test_day_5_2():
    stacks = {1: "ZN", 2: "MCD", 3: "P"}
    stacks = day_5.get_all_stacks(stacks)
    inputs = load_file('tests/test_data/data_2022_5.txt')
    results = day_5.part_2(stacks, inputs)
    assert results == "MCD"

def test_build_queue():
    st = "bvwbjplbgvbhsrlpgdmjqwftvncz"
    queue = Queue()
    queue.build_queue(st)
    assert len(queue) == len(st)
    assert queue.dequeue() == "b"
    assert queue.dequeue() == "v"

def test_cache_queue():
    # This is an overlaoded Queue class where we enforce the max queue length
    tmp_str = "abcdef"
    cache = Cache(max_length=4) 
    X = [cache.enqueue(char) for char in tmp_str]
    assert X == [None, None, None, None, "a", "b"]


@pytest.mark.parametrize("inputs,expected", [("bvwbjplbgvbhsrlpgdmjqwftvncz", 5),
                                             ("nppdvjthqldpwncqszvftbrmjlhg", 6),
                                             ("nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg", 10),
                                             ("zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw", 11),
                                             ])
def test_day_6_1(inputs, expected):
    queue = Queue()
    queue.build_queue(inputs)
    result = day_6.part_1(queue)
    assert result == expected
    
@pytest.mark.parametrize("inputs,expected", [("mjqjpqmgbljsphdztnvjfqwrcgsmlb", 19),
                                             ("bvwbjplbgvbhsrlpgdmjqwftvncz", 23),
                                             ("nppdvjthqldpwncqszvftbrmjlhg", 23),
                                             ("nznrnfrfntjfmvfwmzdfjlvtqnbhcprsg", 29),
                                             ("zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw", 26),
                                             ])
def test_day_6_2(inputs, expected):
    queue = Queue()
    queue.build_queue(inputs)
    result = day_6.part_2(queue)
    assert result == expected
    
    
    