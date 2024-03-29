# solutions/year_2023/day_15.py
from collections import defaultdict
from typing import Dict, List, Optional, Union

from logger_config import logger
from utils import Timer


def parse_input(file_content: str) -> List[str]:
    return file_content.strip("\n").split(",")


def hash_function(x: str, current_value: int = 0) -> int:
    ascii_value = ord(x)
    current_value = ((current_value + ascii_value) * 17) % 256
    return current_value


def part1(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 1 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 1"):
        steps = parse_input(input_data[0])
        current = 0
        for step in steps:
            temp = 0
            for char in step:
                temp = hash_function(char, current_value=temp)
            current += temp
            logger.debug(f"Step: {step} current: {current}")
        return current


# Define a simple Linked List
class Node:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.next = None


def add_to_linked_list(box, name, value):
    if box is None:
        box = Node(name=name, value=int(value))
    else:
        current = box
        # If the current box has the same name, replace with the new value
        if current.name == name:
            current.value = value
            return box
        while current.next is not None:
            current = current.next
        current.next = Node(name=name, value=int(value))
    return box


def remove_from_linked_list(box, name):
    # If the list is empty, there's nothing to remove
    if box is None:
        return None

    # If the head is the node to be removed
    if box.name == name:
        return box.next

    # For all other nodes, keep track of the current node and its predecessor
    current = box
    while current.next is not None:
        if current.next.name == name:
            # If the node to be removed is found, skip it in the list
            current.next = current.next.next
            return box
        current = current.next

    # If the node to be removed was not found, return the original list
    return box


def part2(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 2 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 2 Dicts only:"):
        boxes = defaultdict(dict)  # type: Dict[int, Dict[str, int]]
        steps = parse_input(input_data[0])
        for step in steps:
            if "=" in step:
                key, value = step.split("=")
                hash = 0
                for char in key:
                    hash = hash_function(char, current_value=hash)
                boxes[hash][key] = int(value)

            elif "-" in step:
                key, _ = step.split("-")
                hash = 0
                for char in key:
                    hash = hash_function(char, current_value=hash)
                boxes[hash].pop(key, None)
    return sum(
        (i + 1) * (j + 1) * l for i in boxes for j, l in enumerate(boxes[i].values())
    )
