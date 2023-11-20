import logging
from utils import load_file, Queue
import re
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def get_max_flow(graph):
    total = 0
    for node in graph.nodes.keys():
        total += graph.nodes[node].flow
    return total

class Graph():
    def __init__(self) -> None:
        self.nodes = {}
        self.clock = 0
            
    def test_graph(self):
        graph_keys = self.nodes.keys()
        children_keys = []
        for key in graph_keys:
            children_keys.append(self.nodes[key].children)
        children_keys = [item for sublist in children_keys for item in sublist]
        assert sorted(list(graph_keys)) == sorted(list(set(children_keys)))
        
    
class Node():
    def __init__(self, name, flow, children) -> None:
        self.name = name
        self.flow = flow
        self.children = children
        self.open = False
        self.opened_at = None
        self.visited = 0
    
    

class Path:
    def __init__(self, path, max_flow) -> None:
        self.path = path
        self.clock = 0
        self.total_flow_rate = 0
        self.total_flow = 0
        self.max_flow = max_flow
        self.valves_on = []
    
    def __cmp__(self, other):
        return self.total_flow_rate < other.total_flow_rate
    
    def step_total_flow(self):
        self.total_flow += self.total_flow_rate
    
    def vist(self, node):
        self.clock += 1
        self.step_total_flow()
        if node.flow > 0 and node.name not in self.valves_on:  # This is a slow lookup in a list O(n), not O(1)
            self.clock += 1
            self.step_total_flow()
            # Add the node flow to the total flow
            self.valves_on.append(node.name)
            self.total_flow_rate += node.flow
    
    def check_total_flow(self):
        # The simple breadth first is really slow
        # So we check if the max flow rate has been found yet, then fast forward
        if self.total_flow_rate == self.max_flow:
            import ipdb; ipdb.set_trace()
            while self.clock <= 30:
                self.self.clock += 1
                self.step_total_flow()



def parse_graph_line(line):
    children = re.findall(r"[A-Z]{2}", line)
    node = children.pop(0)
    flow = int(re.findall(r'-?\d+', line)[0])
    return node, flow, children

def build_graph(inputs):
    # We can tell
    graph = Graph()
    for line in inputs:
        node, flow, children = parse_graph_line(line)
        graph.nodes[node] = Node(node, flow, children)
    # Run a little test to make sure all keys are in both graph.nodes and node.children
    graph.test_graph()
    return graph


def breadth_first_search(graph: Graph, root: str, max_flow: int):
    # breadth first search is better for all paths up to a certain length
    # Queue is going to be of the node names
    queue = Queue()
    graph.nodes[root].visited += 1
    queue.enqueue(Path([root], max_flow))
    thirty_min_paths = []
    counter = 0
    while len(queue) > 0:
        if len(queue) > 1000:
            tmp = list(queue.queue)
            tmp = sorted(tmp, key=lambda x: x.total_flow, reverse=True)
            queue.queue.clear()
            queue.build_queue(tmp[:500])
            # import ipdb; ipdb.set_trace()
        path = queue.dequeue()
        counter += 1
        if path.clock == 31 :
            thirty_min_paths.append(path)
            continue
        # Here we get the last node in the path
        if counter % 10000 == 0:
            print(f"{counter} --> queue len {len(queue)}")
        node = path.path[-1]
        path.vist(graph.nodes[node])
        # clock = graph.nodes[node].visit(clock)
        for next_node in graph.nodes[node].children:
            # if graph.nodes[next_node].visited == False:
            new_path = deepcopy(path)
            # if next_node not in new_path.path:
            new_path.path.append(next_node)
            # graph.nodes[next_node].visited = True
            queue.enqueue(new_path)
    print(len(thirty_min_paths))
    
    print(sorted(thirty_min_paths, key=lambda x: x.total_flow, reverse=True)[0].total_flow) 
    # import ipdb; ipdb.set_trace()
    pass        

def part_1(inputs):
    # inputs = ["Valve AA has flow rate=0; tunnels lead to valves DD, II, BB",
    #             "Valve BB has flow rate=13; tunnels lead to valves CC, AA",
    #             "Valve CC has flow rate=2; tunnels lead to valves DD, BB",
    #             "Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE",
    #             "Valve EE has flow rate=3; tunnels lead to valves FF, DD",
    #             "Valve FF has flow rate=0; tunnels lead to valves EE, GG",
    #             "Valve GG has flow rate=0; tunnels lead to valves FF, HH",
    #             "Valve HH has flow rate=22; tunnel leads to valve GG",
    #             "Valve II has flow rate=0; tunnels lead to valves AA, JJ",
    #             "Valve JJ has flow rate=21; tunnel leads to valve II",
    #         ]
    graph = build_graph(inputs)

    max_flow = get_max_flow(graph)
    breadth_first_search(graph, "AA", max_flow)
   

def part_2(inputs):
    pass

def control():
    inputs = load_file("year_2022/data/data_16.txt")
    part_1(inputs)
    part_2(inputs)