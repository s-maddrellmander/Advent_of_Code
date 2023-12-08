# solutions/year_2023/day_08.py
from itertools import cycle
from math import lcm
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def parse_input(input_data: List[str]) -> Tuple[Dict[str, Dict[str, str]], str]:
    """
    Parse the input data into a usable format.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Tuple[Dict[str, Dict[str, str]], str]: The parsed input data.
    """
    instructions = input_data[0]
    nodes = [node.split(" = ") for node in input_data[2:]]
    # pass
    graph = {}
    for node in nodes:
        if node[0] not in graph:
            left, right = node[1].strip("()").split(", ")
            graph[node[0]] = {"left": left, "right": right}
        else:
            raise ValueError("Node already in graph")

    return graph, instructions


def get_all_start_and_end_nodes(
    graph: Dict[str, Dict[str, str]]
) -> Tuple[List[str], List[str]]:
    """
    Get all the start and end nodes in the graph.

    Args:
        graph (Dict[str, Dict[str, str]]): The graph to traverse.

    Returns:
        Tuple[List[str], List[str]]: The start and end nodes.
    """
    # pass
    start_nodes = []
    end_nodes = []
    for node in graph.keys():
        if node[-1] == "A":
            start_nodes.append(node)
        if node[-1] == "Z":
            end_nodes.append(node)
    return start_nodes, end_nodes


def traverse_graph(graph: Dict[str, Dict[str, str]], instructions: str) -> List[str]:
    """
    Traverse the graph according to the instructions.

    Args:
        graph (Dict[str, Dict[str, str]]): The graph to traverse.
        instructions (str): The instructions to follow.

    Returns:
        str: The path taken through the graph.
    """
    # pass
    path = []
    current_node = "AAA"
    instructions_cycle = instructions * 100
    # import ipdb; ipdb.set_trace()

    for direction in instructions_cycle:
        # TODO: infinite loop of instructions
        if direction == "R":
            current_node = graph[current_node]["right"]
        elif direction == "L":
            current_node = graph[current_node]["left"]
        else:
            raise ValueError("Invalid direction")
        path.append(current_node)
        if current_node == "ZZZ":
            break

    return path


def multiple_traversal(
    graph: Dict[str, Dict[str, str]],
    starts: List[str],
    ends: List[str],
    instructions: str,
) -> int:
    """
    Traverse the graph in parallel with multiple starting and ending points.
    Return the number of steps taken until all end points are reached and the same time

    Do this by finding the lowest common multipier for each path.
    """
    instructions_cycle = instructions * 100000
    steps = [0] * len(starts)
    counter = 0
    current_nodes = starts
    logger.info(f"Starts: {starts}")
    logger.info(f"Ends: {ends}")
    logger.info(f"Number of starts: {len(starts)}")
    for direction in instructions_cycle:
        counter += 1
        if counter % 100000 == 0:
            logger.info(
                f"Step: {counter} Current nodes: {current_nodes} steps taken: {steps}"
            )
        for i, node in enumerate(current_nodes):
            if steps[i] == 0:
                if direction == "R":
                    current_nodes[i] = graph[node]["right"]
                elif direction == "L":
                    current_nodes[i] = graph[node]["left"]
                else:
                    raise ValueError("Invalid direction")

                if current_nodes[i][-1] == "Z":
                    steps[i] = counter
                    logger.info(f"Node {i} reached end in {counter} steps")
        if all(x > 0 for x in steps):
            break

    logger.info(f"Steps: {steps}")
    # This is a really important lesson - loops are super expensive, but cycles are predictable and fast
    return lcm(*steps)


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
        # Your solution for part 1 goes here
        graph, instructions = parse_input(input_data)
        path = traverse_graph(graph, instructions)
        steps = len(path)
        return steps


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
        graph, instructions = parse_input(input_data)
        starts, ends = get_all_start_and_end_nodes(graph)
        steps = multiple_traversal(graph, starts, ends, instructions)
        return steps
