import pytest
from utils import load_file, Queue, Cache, TreeNode
import jax.numpy as jnp
import numpy as np
from collections import deque
from tqdm import tqdm
import cmath

from year_2022 import (day_1, day_2, day_3, day_4, day_5, day_6, day_7, day_8, 
                       day_9, day_10, day_11, day_12)


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
    assert jnp.sum(pad_arr[:, 0]) == -5
    assert jnp.sum(pad_arr[0, :]) == -7

def test_get_views():
    arr = jnp.array([[3,0,3,7,3],
		[2,5,5,1,2],
		[6,5,3,3,2],
		[3,3,5,4,9],
		[3,5,3,9,0]])
    arr = day_8.pad_jnp_array(arr)
    probe = (1, 1)
    views = day_8.get_views(probe, arr)
    assert views["up"] == jnp.array([-1])
    assert jnp.allclose(views["right"], jnp.array([0, 3, 7, 3, -1]))
    assert views["node_value"] == 3
    probe = (3, 3)
    views = day_8.get_views(probe, arr)
    assert jnp.allclose(views["up"], jnp.array([5, 3, -1]))
    assert jnp.allclose(views["right"], jnp.array([3, 2, -1]))
    assert views["node_value"] == 3

    count = day_8.loop_all_nodes(arr)
    assert count == 21


@pytest.mark.parametrize("inputs,threshold,expected", [([0, 1, 2, 3, 4], 1, 2),
                                                       ([5, 1, 2, 7, 4], 5, 1),
                                                       ([0, 1, 2, 7, 4], 5, 4),
                                                       ([5, 7, 2, 4, 7, 1, 1, 1, 3, 9], 8, 10),
                                                       ([3, 6, 2, 4, 7, 1, 1, 1, 3, 9], 7, 5),
                                                       ([-1], 3, 1),
                                                       ])
def test_find_first_greater_than(inputs, threshold, expected):
    # Test the function
    numbers = jnp.array(inputs)
    first_greater = day_8.find_first_greater_than(numbers, threshold)
    assert first_greater == expected 


def test_multiply_distances():
    x = [1, 2, 3]
    result = day_8.multiple_distances(x)
    assert result == 6

@pytest.mark.parametrize("node,val,score", [((3, 3), 3, 1),
                                            ((1, 1), 3, 4),
                                            ((4, 3), 5, 8),
                                            ])
def test_day_8_2_indiv_point(node, val, score):
    arr = jnp.array([[3,0,3,7,3],
		[2,5,5,1,2],
		[6,5,3,3,2],
		[3,3,5,4,9],
		[3,5,3,9,0]])
    arr = day_8.pad_jnp_array(arr)
    # The probe needs to account for the padding
    i, j = node
    view = day_8.get_views((i, j), arr)
    assert view["node_value"] == val
    distances = day_8.get_view_length(view)
    dist_prod = day_8.multiple_distances(distances)
    assert dist_prod == score
    


def test_day_8_2():
    arr = jnp.array([[3,0,3,7,3],
		[2,5,5,1,2],
		[6,5,3,3,2],
		[3,3,5,4,9],
		[3,5,3,9,0]])
    arr = day_8.pad_jnp_array(arr)
    score = day_8.get_best_view(arr)
    assert score == 16


def test_make_head_and_tail():
    head = day_9.Node((0, 0))
    assert head.current == (0, 0)
    assert head.history[(0, 0)] == 1
    # Insert tail in head
    tail = day_9.Node((0, 0))
    tail.prev_node = head
    head.next_node = tail
    assert tail.current == (0, 0)
    assert tail.prev_node == head
    assert head.next_node == tail


@pytest.mark.skip()
@pytest.mark.parametrize("kernel,head_loc,tail_loc", [((3, 0), (3, 0), (2, 0)),
                                              ((0, 4), (0, 4), (0, 3)),
                                              ((0, -4), (0, -4), (0, -3)),
                                              ((-3, 0), (-3, 0), (-2, 0))
                                              ])
def test_node_step(kernel, head_loc, tail_loc):
    head = day_9.Node((0, 0))
    tail = day_9.Node((0, 0))
    head.next_node = tail
    tail.prev_node = head
    # Step = R 4
    head.step(kernel) 
    assert head.current == head_loc
    # print(head.history)
    # print(tail.history)
    assert tail.current == tail_loc

# Day 10
def test_parse_lines():
    inputs = ["noop",
              "addx 3",
              "addx -5"]
    assert day_10.parse_line(inputs[0]) == (0, 1)
    assert day_10.parse_line(inputs[1]) == (3, 2)
    assert day_10.parse_line(inputs[2]) == (-5, 2)

def test_update_register():
    cathode_ray = day_10.CathodRay()
    cathode_ray.update_register(0, 1)
    assert cathode_ray.register  == 1
    assert cathode_ray.register_history  == [1]
    assert cathode_ray.clock  == 1
    assert cathode_ray.clock_history == [1]
    cathode_ray.update_register(3, 2)
    assert cathode_ray.register  == 4
    assert cathode_ray.clock  == 3

def test_day_10_1():
    inputs = load_file("tests/test_data/data_2022_10.txt")
    cathode_ray = day_10.CathodRay()   
    result = day_10.loop_instructions(cathode_ray, inputs)
    # import ipdb; ipdb.set_trace()
    assert cathode_ray.get_register_from_cycle(20) == 21
    assert cathode_ray.get_register_from_cycle(60) == 19
    assert cathode_ray.get_register_from_cycle(100) == 18
    assert cathode_ray.get_register_from_cycle(140) == 21
    assert cathode_ray.get_register_from_cycle(180) == 16
    assert cathode_ray.get_register_from_cycle(220) == 18
    
    index_list = [20, 60, 100, 140, 180, 220]
    expected = [420, 1140, 1800, 2940, 2880, 3960]
    vals = day_10.get_results_from_index(cathode_ray, index_list)
    for v, e in zip(vals, expected):
        assert v == e
    val = sum(vals)
    assert val == 13140


def test_day_10_2():
    inputs = load_file("tests/test_data/data_2022_10.txt")
    cathode_ray = day_10.CathodRay()   
    result = day_10.loop_instructions(cathode_ray, inputs)
    pixels = day_10.get_image(cathode_ray)
    
    expected = "##..##..##..##..##..##..##..##..##..##.."
    assert pixels[:40] == expected

    expected_full = "##..##..##..##..##..##..##..##..##..##..###...###...###...###...###...###...###.####....####....####....####....####....#####.....#####.....#####.....#####.....######......######......######......###########.......#######.......#######....."
    assert expected_full == pixels


def test_parse_day_11_input():
    inputs = ["Monkey 0:\n",
                "Starting items: 79, 98\n",
                "Operation: new = old * 19\n",
                "Test: divisible by 23\n",
                    "If true: throw to monkey 2\n",
                    "If false: throw to monkey 3\n"]
    monkey = day_11.Monkey(iter(inputs[1:]), main_divisor=1)
    assert monkey.items == [79, 98]
    assert monkey.true_monkey == 2
    assert monkey.false_monkey == 3
    assert monkey.test_divisor == 23
    assert monkey.operation(10) == 190

def test_worry_level():
    inputs = ["Monkey 0:\n",
                "Starting items: 79, 98\n",
                "Operation: new = old * 19\n",
                "Test: divisible by 23\n",
                    "If true: throw to monkey 2\n",
                    "If false: throw to monkey 3\n"]
    monkey = day_11.Monkey(iter(inputs[1:]), main_divisor=3)
    # Test single item 
    item = monkey.get_item()
    assert item == 79
    worry_level = monkey.assess_worry_level(item)
    assert worry_level == 1501
    # Test the main divisor - rounds down
    worry_level = monkey.div_by_main(worry_level)
    assert worry_level == 500


@pytest.mark.parametrize("inputs,expected", [(100, 2), (101, 3), (0, 2), (-4, 3)])
def test_monkey_test_item(inputs, expected):
    inp_str = ["Monkey 0:\n",
                "Starting items: 79, 98\n",
                "Operation: new = old * 19\n",
                "Test: divisible by 5\n",
                    "If true: throw to monkey 2\n",
                    "If false: throw to monkey 3\n"]
    monkey = day_11.Monkey(iter(inp_str[1:]), main_divisor=3)
    assert monkey.true_monkey == 2
    assert monkey.false_monkey == 3
    next_id = monkey.test_item(inputs)
    assert next_id == expected

def test_throw_to_monkey():
    inp_str = ["Monkey 0:\n",
                "Starting items: 5, 17\n",
                "Operation: new = old * 19\n",
                "Test: divisible by 5\n",
                    "If true: throw to monkey 1\n",
                    "If false: throw to monkey 2\n"]
    monkey_0 = day_11.Monkey(iter(inp_str[1:]), main_divisor=3)
    inp_str = ["Monkey 1:\n",
                "Starting items: 10, 920\n",
                "Operation: new = old * 10\n",
                "Test: divisible by 1\n",
                    "If true: throw to monkey 2\n",
                    "If false: throw to monkey 3\n"]
    monkey_1 = day_11.Monkey(iter(inp_str[1:]), main_divisor=3)
    inp_str = ["Monkey 2:\n",
                "Starting items: 10, 920\n",
                "Operation: new = old * 10\n",
                "Test: divisible by 5\n",
                    "If true: throw to monkey 2\n",
                    "If false: throw to monkey 3\n"]
    monkey_2 = day_11.Monkey(iter(inp_str[1:]), main_divisor=3)
    monkeys = [monkey_0, monkey_1, monkey_2]
    monkeys = day_11.turn(monkeys, 0)
    assert monkeys[0].has_items() == False
    assert monkeys[1].items[-1] == 920
    assert monkeys[2].items[-1] == 107
   
def test_day_11_1():
    filename = "tests/test_data/data_2022_11.txt"
    with open(filename) as f:
        inputs = f.read()
    monkeys = [day_11.Monkey(iter(monkey.split("\n")[1:]), 3) for monkey in inputs.split("\n\n")]
    monkeys = day_11.round(monkeys)
    assert monkeys[0].items == [20, 23, 27, 26]
    assert monkeys[1].items == [2080, 25, 167, 207, 401, 1046]
    monkeys = day_11.round(monkeys)
    assert monkeys[0].items == [695, 10, 71, 135, 350]
    assert monkeys[1].items == [43, 49, 58, 55, 362]
    for _ in range(18):
        monkeys = day_11.round(monkeys)
    assert monkeys[0].items == [10, 12, 14, 26, 34]
    assert monkeys[1].items == [245, 93, 53, 199, 115]
    
    assert day_11.get_counters(monkeys) == [101, 95, 7, 105]
    assert day_11.top_k_mul(day_11.get_counters(monkeys), 2) == 10605

@pytest.mark.parametrize("num_rounds,expected", [(1, 24), (20, 10197),
                                                 (1000, 5204*5192),
                                                 (2000, 10419*10391), (3000, 15638*15593),
                                                 (5000, 26075*26000)
                                                ])
def test_day_11_2(num_rounds, expected):
    filename = "tests/test_data/data_2022_11.txt"
    with open(filename) as f:
        inputs = f.read()
    monkeys = [day_11.Monkey(iter(monkey.split("\n")[1:]), 1) for monkey in inputs.split("\n\n")]
    lcm = np.prod([m.test_divisor for m in monkeys])
    # lcm = 1
    monkeys = [day_11.Monkey(iter(monkey.split("\n")[1:]), 1, lcm) for monkey in inputs.split("\n\n")]
    for _ in range(num_rounds):
        monkeys = day_11.round(monkeys) 
    counters = day_11.get_counters(monkeys)
    result = day_11.top_k_mul(counters, 2)
    assert result == expected
    
def test_convert_alpha_to_numeric():
    grid = ["Sabqponm",
            "abcryxxl",
            "accszExk",
            "acctuvwj",
            "abdefghi"]
    parsed, start, end = day_12.convert_alpha_to_numeric(grid)
    assert start == complex(0, 0)
    assert end == complex(2, 5)
    assert parsed[complex(4, 7)] == 8
    
def test_get_start_end():
    grid = ["Sabqponm",
            "abcryxxl",
            "accszExk",
            "acctuvwj",
            "abdefghi"]
    start, end = day_12.get_start_end(grid)
    assert start == complex(0, 0)
    assert end == complex(2, 5)

def test_map_options():
    _map = {complex(0, 0):1, complex(0, 1): 2, complex(1, 1): 3}
    climber = day_12.Climber(start=complex(0,0), end=complex(10,10), map=_map)
    options = climber.get_options()
    assert options == [complex(0, 1)]


def test_bfs_test_graph():
    grid = ["Sabqponm",
            "abcryxxl",
            "accszExk",
            "acctuvwj",
            "abdefghi"]
    parsed, start, end = day_12.convert_alpha_to_numeric(grid)
    bfs = day_12.Climber(start, end, parsed)
    path = bfs.bfs_shortest_path(start, end)
    assert len(path) - 1 == 31

def test_day_12_2():
    grid = ["Sabqponm",
            "abcryxxl",
            "accszExk",
            "acctuvwj",
            "abdefghi"]
    result = day_12.part_2(grid)
    assert result == 29 