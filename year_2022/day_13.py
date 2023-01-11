import logging
from utils import load_file

"""
Packet comparison.
- If both values are integers, the lower integer should come first.
- If both values are lists, compare the first value of each list, then the
  second value, and so on. If the left list runs out of items first,
  the inputs are in the right order.
"""

def parse_line(line) -> list:
    return eval(line)

def list_pop(test):
    try:
        return test.pop(0)
    except IndexError:
        return False

def check_length(left, right):
    if left is [] and right is not []:
        return True
    elif left is not [] and right is []:
        return False
    else:
        return None
    
def corall_type(left, right):
    if type(left) == type(right):
        return (left, right)
    elif type(left) == list and type(right) == int:
        return (left, [right])
    elif type(right) == list and type(left) == int:
        return ([left], right)
    
    
def compare_list(x, y):
    if isinstance(x, int) and isinstance(y, int):
        if x == y:
            return None 
        return (x < y)

    if type(x) is list and type(y) is list:
        for a, b in zip(x, y):
            sub_check = compare_list(a, b)
            if sub_check is not None:
                return sub_check
        return compare_list(len(x), len(y))
    if isinstance(x, int) and isinstance(y, list):
        x = [x]
        return compare_list(x, y)
    elif isinstance(x, list) and isinstance(y, int):
        y = [y]
        return compare_list(x, y)
    
    # return True
    

def part_1(inputs):
    # Loop through and pass in pairs
    inputs = [inp for inp in inputs if inp != ""]
    index_tracker = []
    for index, i in enumerate(range(0, len(inputs), 2), 1):
        left = parse_line(inputs[i])
        right= parse_line(inputs[i + 1])
        check = compare_list(left, right)
        if check: index_tracker.append(index)
    result = sum(index_tracker)
    logging.info(f"Part 1: {result}")
    return result, index_tracker

def part_2(inputs):
    inputs = [inp for inp in inputs if inp != ""]
    index_tracker = [parse_line(inputs[0])]
    inputs.append("[[2]]")
    inputs.append("[[6]]")
    for item in inputs[1:]:
        state = False
        item = parse_line(item)
        for i in range(len(index_tracker)):
            state = compare_list(item, index_tracker[i])
            if state == True:
                index_tracker.insert(i, item)
                break
        if state == False:
            index_tracker.insert(i+1, item)
    div_1 = index_tracker.index([[2]])
    div_2 = index_tracker.index([[6]])
    res = (div_1+1) * (div_2+1)
    logging.info(f"Part 2: {res}")
    return res, index_tracker 
                
        

def control():
    inputs = load_file("year_2022/data/data_13.txt")
    part_1(inputs)
    part_2(inputs)