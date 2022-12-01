from utils import load_file, Timer
import logging
import torch

def get_groups(stream):
    # Want a finite state machine or something here
    state = 0
    garbage = False
    total_groups = 0
    score = 0
    cancel = 0
    garb_counter = 0
    for x in stream:
        if cancel > 0:
            cancel -= 1
            continue
        if x == "!":
            cancel = 1
            continue
        if garbage is False:
            if x == "{":
                state += 1
            if x == "}":
                score += state
                state -= 1
                total_groups += 1
            if x == "<":
                garbage = True
        else:
            if x == ">":
                garbage = False
            else:
                garb_counter += 1
    return score, total_groups, garb_counter

def part_1(inputs):
    score, total_groups, _ = get_groups(inputs)
    logging.info(f"Total Score {score}")
    return score

def part_2(inputs):
    score, total_groups, garb_counter = get_groups(inputs)
    logging.info(f"Total Garb  {garb_counter}")
    return garb_counter

def control():
    inputs = load_file('year_2017/data/data_9.txt')[0]
    with Timer('Part 1'):
        part_1(inputs)
    with Timer('Part 2'):
        part_2(inputs)