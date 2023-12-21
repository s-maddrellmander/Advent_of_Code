# solutions/year_2023/day_00.py
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from logger_config import logger
from utils import Timer


def parse_string_to_values(input_str):
    input_str = input_str.strip("{}")
    pairs = input_str.split(",")
    values = [int(pair.split("=")[1]) for pair in pairs]
    values = {k: v for k, v in zip(["x", "m", "a", "s"], values)}
    return values


def parse_string_to_nodes(input_str):
    # vr{x>269:A,x<163:A,s<3259:R,A}
    name, rest = input_str.split("{")
    name = name.strip()
    rest = rest.strip("}")
    conditions = rest.split(",")
    conditions = [condition.split(":") for condition in conditions]
    return (name, conditions)


def parse_input(input_data: List[str]):
    # Find the index of the empty string
    split_index = input_data.index("")
    # Parse the first section into nodes:
    parse_nodes = input_data[:split_index]
    # assert "sg{x>1447:msc,bsb}" in nodes[-1]
    values = input_data[split_index + 1 :]
    # assert "{x=238,m=232,a=127,s=30}" in values[0]

    nodes: Dict[str, Any] = {
        name: conditions
        for name, conditions in [parse_string_to_nodes(node) for node in parse_nodes]
    }
    values = [parse_string_to_values(value) for value in values]
    return nodes, values


def evaluate_conditions(conditions, values_dict):
    for condition in conditions:
        if len(condition) == 1:
            return condition[0]

        # Split the condition and the return string
        cond_str, return_str = condition
        # Parse the condition
        if "<" in cond_str:
            var, threshold = cond_str.split("<")
            if values_dict[var] < int(threshold):
                return return_str
        elif ">" in cond_str:
            var, threshold = cond_str.split(">")
            if values_dict[var] > int(threshold):
                return return_str
    return 0


def calculate_value(xmas, nodes):
    output = None
    current = "in"
    while output not in ["A", "R"]:
        output = evaluate_conditions(nodes[current], xmas)
        logger.debug(f"Current: {current}, Output: {output}")
        if output == 0:
            raise ValueError("No output found")
        current = output
    return output


def score_xmas(xmas: Dict) -> int:
    return sum(xmas.values())


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
        score = 0
        nodes, values = parse_input(input_data)
        for value in values:
            output = calculate_value(value, nodes)
            logger.debug(f"Output: {output}")
            if output == "A":
                score += score_xmas(value)
        return score


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

    with Timer("Part 2"):
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
