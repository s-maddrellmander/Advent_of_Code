# solutions/year_2024/day_23.py

from logger_config import logger
from utils import Timer
from itertools import combinations

def parse_lan(data: list[str]) -> dict[str, set[str]]:
    # Parse the input data into a graph 
    lan_party: dict[str, set[str]] = {}
    for line in data:
        a, b = line.split("-")
        if a not in lan_party:
            lan_party[a] = set()
        if b not in lan_party:
            lan_party[b] = set()
        lan_party[a].add(b)
        lan_party[b].add(a)
    return lan_party


def reverse_lan(lan_party: dict[str, set[str]]) -> dict[str, set[str]]:
    # Reverse the graph
    reversed_lan: dict[str, set[str]] = {}
    for k, v in lan_party.items():
        for n in v:
            if n not in reversed_lan:
                reversed_lan[n] = set()
            reversed_lan[n].add(k)
    return reversed_lan

def print_adjacency_matrix(lan_party: dict[str, set[str]]):
    # Print the adjacency matrix full rank - including missing edges
    nodes = sorted(lan_party.keys())
    num_nodes = len(nodes)
    adjacency_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, node in enumerate(nodes):
        for neighbour in lan_party[node]:
            j = nodes.index(neighbour)
            adjacency_matrix[i][j] = 1
    for row in adjacency_matrix:
        print(row)
    return adjacency_matrix

def find_interconnected_triplets(lan_party: dict[str, set[str]]) -> list[list[str]]:
    # Find all possible triplets
    triplets = set()
    for a in lan_party:
        for b in lan_party[a]:
            for c in lan_party[b]:
                if c in lan_party[a]:
                    if a[0] == "t" or b[0] == "t" or c[0] == "t":
                        triplets.add(frozenset([a, b, c]))
    return triplets
                
def find_max_clique(graph):
    def is_clique(vertices):
        # Check if all vertices are connected to each other
        for v1 in vertices:
            for v2 in vertices:
                if v1 != v2 and v2 not in graph[v1]:
                    return False
        return True
    
    n = len(graph)
    # Try all possible combinations of vertices, from largest to smallest
    for size in range(n, 0, -1):
        print(f"Trying clique of size {size} / {n}", end="\r")
        # Check all combinations of 'size' vertices
        for combo in combinations(graph.keys(), size):
            if is_clique(combo):
                return set(combo)
    


def part1(input_data: list[str] | None) -> str | int:
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
        lan_party = parse_lan(input_data)
        reverse_lan_party = reverse_lan(lan_party)
        # number of nodes, and edges
        logger.debug(f"Number of nodes: {len(lan_party)}")
        logger.debug(f"Number of edges: {sum(len(v) for v in lan_party.values())}")
        triplets = find_interconnected_triplets(lan_party)


    return len(triplets)


    
    return set()  # Empty set if no clique found

def part2(input_data: list[str] | None) -> str | int:
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
        # Finding the biggest clique
        lan_party = parse_lan(input_data)
        max_clique = find_max_clique(lan_party)
        
        return len(max_clique)
        
