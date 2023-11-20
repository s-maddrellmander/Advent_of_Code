import re
import logging
import numpy as np
from tqdm import tqdm 
from copy import deepcopy


from utils import load_file

get_numbers = lambda a: list(map(int, re.findall('\d+', a)))

class Monkey:
    def __init__(self, inputs, main_divisor, lcm=None):
        super().__init__()
        self.items = get_numbers(next(inputs).split(": ")[1])
        self.operation_str = next(inputs).split('= ')[-1]
        self.test_divisor = self.get_number(next(inputs))
        self.true_monkey = self.get_number(next(inputs))
        self.false_monkey = self.get_number(next(inputs))
        self.main_divisor = main_divisor
        self.lowest_common_multiplier = lcm
        self.test_counter = 0
    
    def has_items(self):
        if len(self.items) > 0:
            return True
        else:
            return False
        
    def get_item(self):
        if len(self.items) > 0:
            return self.items.pop(0)  # Takes from the front of the queue
    
    def get_number(self, inp_str):
        return int(inp_str.split(" ")[-1])
    
    def assess_worry_level(self, item):
        worry_level = eval(self.operation_str, {"old": item}) 
        return worry_level
    
    def div_by_main(self, worry_level):
        if self.lowest_common_multiplier == None:
            worry_level = worry_level // self.main_divisor
        else:
            worry_level = worry_level % self.lowest_common_multiplier
        return worry_level
        
    def operation(self, old):
        return eval(self.operation_str)
        
    def test_item(self, item):
        self.test_counter += 1
        if item % self.test_divisor == 0:
            return self.throw(True)
        else:
            return self.throw(False)
    
    def throw(self, outcome):
        logic = {True: self.true_monkey, False: self.false_monkey} 
        return logic[outcome]
    
    def catch(self, item):
        self.items.append(item)
        
def turn(monkeys, id):
    while monkeys[id].has_items():
        item = monkeys[id].get_item()
        item = monkeys[id].assess_worry_level(item)
        item = monkeys[id].div_by_main(item)
        throw_to = monkeys[id].test_item(item)
        monkeys[throw_to].catch(item)
    return monkeys

def round(monkeys):
    for id in range(len(monkeys)):
        monkeys = turn(monkeys, id)
    return monkeys

def get_counters(monkeys):
    return [monkeys[_].test_counter for _ in range(len(monkeys))]

def top_k_mul(counters, k):
    top_k = sorted(counters)[-k:]
    return np.prod(top_k)

def part_1(inputs):
    monkeys = [Monkey(iter(monkey.split("\n")[1:]), 3) for monkey in inputs.split("\n\n")]    
    for _ in range(20):
        monkeys = round(monkeys)
    counters = get_counters(monkeys)
    result = top_k_mul(counters, 2)
    logging.info(f"Part 1: {result}")
    return result

def part_2(inputs):
    monkeys = [Monkey(iter(monkey.split("\n")[1:]), 1) for monkey in inputs.split("\n\n")]
    lcm = np.prod([m.test_divisor for m in monkeys])
    monkeys = [Monkey(iter(monkey.split("\n")[1:]), 1, lcm) for monkey in inputs.split("\n\n")] 
    for _ in tqdm(range(10000)):
        monkeys = round(monkeys)
    counters = get_counters(monkeys)
    result = top_k_mul(counters, 2)
    logging.info(f"Part 2: {result}")
    return result

def control():
    filename = "year_2022/data/data_11.txt"
    with open(filename) as f:
        inputs = f.read()
    part_1(deepcopy(inputs))
    part_2(deepcopy(inputs))