from utils import load_file, Timer
import logging

def parse_input_to_graph(input):
    """Assume nodes create directed graph."""
    graph = {}
    for line in input:
        # Split on `->` first
        line = line.split("->")
        line[0] = line[0].replace("(", "")
        line[0] = line[0].replace(")", "")
        base_node = line[0].split(" ")
        if base_node[0] not in graph:
            graph[base_node[0]] = dict(value=int(base_node[1]), out=[], rec=[])
        else:
            graph[base_node[0]]["value"] = int(base_node[1])
            # if "out" not in graph[base_node[0]].keys():
            #     graph[base_node[0]]["out"] = []
        if len(line) > 1:
            line[1] = line[1].replace(" ", "")
            out_nodes = line[1].split(",")
            for node in out_nodes:
                graph[base_node[0]]["out"].append(node)
            # Add the reciever nodes
            for node in out_nodes:
                if node not in graph:
                    graph[node] = dict(rec=[base_node[0]], out=[], value=None)
                else:
                    if base_node[0] not in graph[node]["rec"]: 
                        graph[node]["rec"].append(base_node[0])
    return graph


def part_1(graph):
    # We trust the graph is tree like with no loops
    visited = []
    # Random starting node
    location = [list(graph.keys())[0]]
    while len(location) > 0:
        current = location.pop(0)
        next_node = graph[current]["rec"]
        location.extend(next_node)
    logging.info(f"Head node {current}")
    return current

def part_2(graph, base_node):
    """Balance the subtowers - there's only one weight off
    - 3x trees - ones of the sums is different to the others
    """
    visited = set() # Set to keep track of visited nodes.

    def dfs(visited, graph, node, runner):
        if node not in visited:
            runner += graph[node]["value"]
            values = [graph[n_out]["value"] for n_out in graph[node]["out"]]
            if len(set(values)) > 1:
                print("ARgh")
            visited.add(node)
            for neighbour in graph[node]["out"]:
                runner = dfs(visited, graph, neighbour, runner)
        return runner

    sums = [dfs(visited, graph, bn, 0) for bn in graph[base_node]["out"]]
    assert len(set(sums)) == 2
    diff = int(abs(min(sums) - max(sums)))
    logging.info(f"Part 2: The difference is {diff}")

    return diff, sums

def control():
    # input = load_file("year_2017/data/data_7.txt")
    input = load_file("tests/test_data/data_2017_7_1.txt")
    input = parse_input_to_graph(input)
    with Timer("Day 7 Part 1"):
        base_node = part_1(input)
    with Timer("Day 7 Part 2"):
        part_2(input, base_node)



