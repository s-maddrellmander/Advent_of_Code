import pytest
from utils import load_file, Queue, Cache, TreeNode
import jax.numpy as jnp
from collections import deque

from year_2022 import (day_1, day_2, day_3, day_4, day_5, day_6, day_7, day_8)


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
    
    
def test_tree_structure(): 
    inst = ["$ cd /", "dir a", "123 file", "dir b", "$ cd b", "123 file"]
    queue = Queue()
    queue.build_queue(inst)
    tree = day_7.build_tree(queue)
    assert tree.name == "/"
    assert tree.sub_tree[0].name == "b"
    assert tree.sub_tree[0].leaves[0].name == "file"
    assert tree.sub_tree[0].leaves[0].size == 123

def test_sum_tree():
    inst = ["$ cd /", "dir a", "123 file", "123 file", "dir b", "$ cd b", "123 file", "$ cd ..", "$ cd a", "123 file"]
    queue = Queue()
    queue.build_queue(inst)
    tree = day_7.build_tree(queue)
    assert tree.tree_size() == 123 + 123 + 123 + 123
    sub = tree.sub_tree[0]
    assert sub.tree_size() == 123

def test_day_7_1():
    inputs = load_file("tests/test_data/data_2022_7.txt")
    result = day_7.part_1(inputs)
    assert result == 95437

def test_day_7_2():
    inputs = load_file("tests/test_data/data_2022_7.txt")
    result = day_7.part_2(inputs)
    assert result == 24933642

def test_day_7_2_lineby():
    inputs = load_file("tests/test_data/data_2022_7.txt")
    queue = Queue()
    
    queue.build_queue(inputs)    
    tree = day_7.build_tree(queue)
    current_total_space = tree.tree_size()
    TOTALSPACE = 70000000
    SPACEREQUIRED = 30000000
    USEDSPACE = current_total_space
    assert current_total_space == sum([int(line.split(" ")[0]) for line in inputs if line.split(" ")[0].isnumeric()])
    assert USEDSPACE == 48381165
    assert TOTALSPACE - USEDSPACE == 21618835
    SPACENEEDEDTOBECLREAED = SPACEREQUIRED - (TOTALSPACE - USEDSPACE)
    assert SPACENEEDEDTOBECLREAED == 8381165
    all_dir_sizes = []
    all_dir_sizes = day_7.check_to_del(tree, all_dir_sizes, target_size=SPACENEEDEDTOBECLREAED)
    all_dir_sizes = [x.tree_size() for x in all_dir_sizes]
    valid_dirs = [x for x in all_dir_sizes if x > SPACENEEDEDTOBECLREAED]
    minner = min(valid_dirs)
    assert minner == 24933642


def test_parse_array_to_jax():
    inputs = ["12345", "66766", "00304"]
    jnp_arr = day_8.parse_array_to_jax(inputs)
    assert jnp_arr.shape == (3, 5)


def test_pad_jax_array():
    arr = jnp.ones((3, 5))
    pad_arr = day_8.pad_jnp_array(arr)
    assert pad_arr.shape == (5, 7)
    assert jnp.sum(pad_arr[:, 0]) == 0
    assert jnp.sum(pad_arr[0, :]) == 0

def test_get_views():
    arr = jnp.array([[3,0,3,7,3],
		[2,5,5,1,2],
		[6,5,3,3,2],
		[3,3,5,4,9],
		[3,5,3,9,0]])
    arr = day_8.pad_jnp_array(arr)
    probe = (1, 1)
    views = day_8.get_views(probe, arr)
    assert views["up"] == jnp.array([0])
    assert jnp.allclose(views["right"], jnp.array([0, 3, 7, 3, 0]))
    assert views["node_value"] == 3
    probe = (3, 3)
    views = day_8.get_views(probe, arr)
    assert jnp.allclose(views["up"], jnp.array([0, 3, 5]))
    assert jnp.allclose(views["right"], jnp.array([3, 2, 0]))
    assert views["node_value"] == 3

    count = day_8.loop_all_nodes(arr)
    assert count == 21
