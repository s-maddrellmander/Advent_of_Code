from utils import load_file, Timer
import logging
import torch

def get_register(inputs):
    regs = {line.split(" ")[0]: 0 for line in inputs}
    return regs

def parse_line(line):
    # Split into reg, instruction, condition
    parsed = line.split(" if ")
    parsed[0] = parsed[0].split(" ")
    inst = int(parsed[0][2]) * (1 if parsed[0][1] == "inc" else -1)
    cond = parsed[1]
    # cond = {"node": cond_str[0], "op": cond_str[1], "val": int(cond_str[2])# cond = {"node": cond_str[0], "op": cond_str[1], "val": int(cond_str[2])}}
    register = {"reg": parsed[0][0],
                "inst": inst,
                "cond": cond}
    return register

def eval_cond(reg, cond):
    # Evaluate the condition on the register
    cond_str = cond.split(" ")
    cond = {"node": cond_str[0], "op": cond_str[1], "val": int(cond_str[2])}
    if cond['node'] not in reg:
        reg[cond['node']] = 0
    eval_str = f"{reg[cond['node']]} {cond['op']} {cond['val']}"
    return eval(eval_str)
    
def part_1(inputs, register):
    for line in inputs:
        line = parse_line(line)
        if eval_cond(register, line['cond']):
            register[line["reg"]] += line["inst"]
    
    max_val = max(register.values())
    logging.info(f"Max val: {max_val}")
    return max_val

def part_2(inputs, register):
    max_val = 0
    for line in inputs:
        line = parse_line(line)
        if eval_cond(register, line['cond']):
            register[line["reg"]] += line["inst"]
    
        _max_val = max(register.values())
        if _max_val > max_val:
            max_val = _max_val
    logging.info(f"Max val: {max_val}")
    return max_val

def control():
    inputs = load_file('year_2017/data/data_8.txt')
    register = get_register(inputs)
    with Timer("Part 1"):
        part_1(inputs, register)
    register = get_register(inputs)
    with Timer("Part 2"):
        part_2(inputs, register)
