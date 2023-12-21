# solutions/year_2023/day_20.py
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx

from logger_config import logger
from utils import Timer


class Node:
    def __init__(self, name: str, in_edges=None, out_edges=None):
        self.name = name
        self.in_edges = in_edges if in_edges else set()
        self.out_edges = out_edges if out_edges else set()
        self.count_low = 0
        self.count_high = 0

    def check_value(self, value):
        # 1 = High, 0 = low, None = unknown
        try:
            value in [0, 1, None]
        except:
            raise ValueError(f"Value must be 0, 1, or None. Got {value}")

        if value == 1:
            self.count_high += 1
        elif value == 0:
            self.count_low += 1

    def process_value(self, value, edge):
        self.check_value(value)
        return value


class Broadcast(Node):
    # This is the start node that sends into the network
    def __init__(self, name, in_edges=None, out_edges=None):
        super().__init__(name, in_edges, out_edges)

    def process_value(self, value, edge):
        return super().process_value(value, edge)


class FlipFlop(Node):
    def __init__(self, name, in_edges=None, out_edges=None):
        super().__init__(name, in_edges, out_edges)
        self.state = False  # off, True = on

    def process_value(self, value, edge):
        self.check_value(value)
        # If high value do nothing and stop propagating
        if value == 1:
            return None
        else:
            if self.state == False:
                # if off FlipFlop Turns on
                self.state = True
                return 1
            else:
                # if on FlipFlop Turns off
                self.state = False
                return 0


class Conjunction(Node):
    def __init__(self, name, in_edges=None, out_edges=None):
        super().__init__(name, in_edges, out_edges)
        self.memory = None

    def process_value(self, value, in_edge):
        self.check_value(value)
        if self.memory is None:
            # populate with all the edges as 0
            logger.info(f"Populating memory for {self.name}")
            self.memory = {_edge: 0 for _edge in self.in_edges}
        # Update the memory with the value
        self.memory[in_edge] = value
        if all([self.memory[edge] == 1 for edge in self.in_edges]):
            return 0
        else:
            return 1


def parse_input_to_graph(input_data: List[str]):
    nodes = {}
    for line in input_data:
        _name, outputs = line.split(" -> ")
        outputs = outputs.split(", ")
        if "&" in _name:
            _name = _name.replace("&", "")
            node = Conjunction(_name, out_edges=set(outputs))
        elif "%" in _name:
            _name = _name.replace("%", "")
            node = FlipFlop(_name, out_edges=set(outputs))
        else:
            node = Broadcast(_name, out_edges=set(outputs))
        nodes[_name] = node
        for out in outputs:
            if out not in nodes:
                nodes[out] = Node(out)
            nodes[out].in_edges.add(_name)
    # Now add all the in edges for everything
    for node in nodes.values():
        for edge in node.out_edges:
            nodes[edge].in_edges.add(node.name)
    return nodes


def draw_graph(nodes):
    G = nx.DiGraph()
    for name, node in nodes.items():
        G.add_node(name)
        for out_edge in node.out_edges:
            G.add_edge(name, out_edge)

    pos = nx.spring_layout(G)  # Use spring layout
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=100,
        alpha=0.6,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        arrowstyle="->",
        arrowsize=20,
    )
    plt.show()


def propagate_value(nodes, start_node_name, initial_value):
    to_process = [(start_node_name, initial_value, None)]
    while to_process:
        # logger.debug(f"to_process: {to_process}")
        current_node_name, value, incoming_edge = to_process.pop(0)

        current_node = nodes[current_node_name]
        processed_value = current_node.process_value(value, incoming_edge)
        # if current_node_name == "vk":
        #     import ipdb; ipdb.set_trace()
        #     if value == 1:
        #         logger.info(f"incoming: {value}, {incoming_edge} -> {current_node.memory}")

        # logger.debug(f"current_node: {current_node_name}, value: {value}, processed_value: {processed_value}")
        if processed_value is None:
            continue
        # Pass the processed value to each connected node
        for out_edge in current_node.out_edges:
            to_process.append((out_edge, processed_value, current_node_name))

    node_values = {
        name: node.state for name, node in nodes.items() if isinstance(node, FlipFlop)
    }
    return node_values


def count_counts(nodes):
    details = {name: (node.count_low, node.count_high) for name, node in nodes.items()}
    high = sum([node.count_high for node in nodes.values()])
    low = sum([node.count_low for node in nodes.values()])
    return low, high, details


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
        nodes = parse_input_to_graph(input_data)
        # draw_graph(nodes)
        for button_press in range(1000):
            propagate_value(nodes, "broadcaster", 0)
        low, high, details = count_counts(nodes=nodes)
        return low * high


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
        nodes = parse_input_to_graph(input_data)
        # draw_graph(nodes)
        step = 0
        vk_step = ks_step = dl_step = pm_step = 0
        # while nodes['rx'].count_low == 0:
        while any([vk_step == 0, ks_step == 0, dl_step == 0, pm_step == 0]):
            propagate_value(nodes, "broadcaster", 0)
            import ipdb

            ipdb.set_trace()
            if nodes["dt"].memory["vk"] == 1:
                vk_step = step
                logger.info(f"vk_step: {step}")
            if nodes["dt"].memory["ks"] == 1:
                ks_step = step
                logger.info(f"ks_step: {step}")
            if nodes["dt"].memory["dl"] == 1:
                dl_step = step
                logger.info(f"dl_step: {step}")
            if nodes["dt"].memory["pm"] == 1:
                pm_step = step
                logger.info(f"pm_step: {step}")
            if step % 100000 == 0:
                logger.info(
                    f"({step}) Node rx: {nodes['rx'].count_low}, Node rx: {nodes['rx'].count_high}"
                )
            # logger.info(f"{vk_step}, {ks_step}, {dl_step}, {pm_step}")

            step += 1
        return -1
