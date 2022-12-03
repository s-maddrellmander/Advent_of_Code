from utils import load_file
import logging
import string


def split_lists_in_half(inputs):
    length = len(inputs) // 2
    comp_1 = inputs[:length]
    comp_2 = inputs[length:]
    assert len(comp_1) == len(comp_2)
    return comp_1, comp_2 

def find_double_char(comp_1, comp_2):
    common_characters = ''.join(set(comp_1).intersection(comp_2))
    return common_characters

def score_char(char):
    for i, x in enumerate(string.ascii_letters, 1):
        if char == x:
            return i

def part_1(inputs):
    score = 0
    for line in inputs:
        comp_1, comp_2 = split_lists_in_half(line)
        char = find_double_char(comp_1, comp_2)
        score += score_char(char)
    logging.info(f"Part 1 {score}") 
    return score
        

def part_2(inputs):
    # More tricky, common item for every three lines. Should just be able to use 
    # The functions already declared. 
    score = 0
    for i in range(0, len(inputs), 3):
        lines = inputs[i:i+3]
        combs = [find_double_char(lines[i], lines[j]) for i, j in [[0, 1], [1, 2], [0, 2]]]
        combs = [find_double_char(combs[0], combs[1]), find_double_char(combs[1], combs[2])]
        char = find_double_char(combs[0], combs[1])
        score += score_char(char) 
    logging.info(f"Part 2 {score}") 
    return score

def control():
    inputs = load_file("year_2022/data/data_3.txt") 
    part_1(inputs)
    part_2(inputs)