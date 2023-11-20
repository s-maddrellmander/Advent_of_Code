import logging
from utils import load_file
from collections import deque
import re

"""
This is a good example of how to use stacks
"""

def build_stack(inputs):
    # Str to list
    inputs = [x for x in inputs]
    stack = deque(inputs)
    return stack

def get_all_stacks(inputs):
    stacks = {x: build_stack(inputs[x]) for x in inputs.keys()}
    return stacks 
    
def part_1(stacks, inputs):
    # Start from the actual instructions
    inputs = inputs[inputs.index("") + 1:]
    for inst in inputs:
        move = [int(x) for x in re.findall(r'\d+', inst)]
        for _ in range(move[0]):
            stacks[move[2]].append(stacks[move[1]].pop())
    res = "".join(stacks[_].pop() for _ in stacks.keys())
    logging.info(f"Part 1: {res}")
    return res
    

def part_2(stacks, inputs):
    # Start from the actual instructions
    inputs = inputs[inputs.index("") + 1:]
    for inst in inputs:
        move = [int(x) for x in re.findall(r'\d+', inst)]
        queue = deque()
        for _ in range(move[0]):
            if len(stacks[move[1]]) > 0:
                queue.append(stacks[move[1]].pop())
        while len(queue) > 0:
            stacks[move[2]].append(queue.pop())
    res = "".join(stacks[_].pop() for _ in stacks.keys())
    logging.info(f"Part 1: {res}")
    return res


def control():
    inputs = load_file('year_2022/data/data_5.txt')
    stack_lists = {1: "SMRNWJVT", 2: "BWDJQPCV", 3: "BJFHDRP", 4: "FRPBMND", 5: "HVRPTB",
                   6: "CBPT", 7: "BJRPL", 8: "NCSLTZBW", 9: "LSG"}
    stacks = get_all_stacks(stack_lists)
    assert len(stacks[1]) == 8
    part_1(stacks, inputs)
    # Reset the stacks for part 2
    stacks = get_all_stacks(stack_lists)
    part_2(stacks, inputs)