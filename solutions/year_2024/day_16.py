# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer

"""
(1) Tuen the input into a graph 
(2) Each node links to the valid neigbours 
(3) When "moving" we can for cost 1 move in the direction we are currently going, or +1000 to change direction
(4) Find the lowest cost path from start to the end. 
"""

def map_to_graph(input_data: list[str]) -> dict[complex, list[complex]]:
    graph = {}
    start = None
    end = None
    for y in range(len(input_data)):
        for x in range(len(input_data[y])):
            if input_data[y][x] == "#":
                continue
            current = complex(x, y)
            graph[current] = []
            if input_data[y][x] == "S":
                start = current
            elif input_data[y][x] == "E":
                end = current
    for node in graph:
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new = node + complex(dx, dy)
            if new in graph:
                graph[node].append(new)
    return graph, start, end


def move_cost(current: complex, next: complex, direction: complex, part2=False) -> int:
    if next == current + direction:
        return 1
    return 1001 # 1000 to change direction, 1 to move

def dikstra(graph: dict[complex, list[complex]], start: complex, end: complex) -> int:
    cost = {(node, _dir): float('inf') for _dir in [-1, 1, 1j, -1j] for node in graph}
    direction = 1 + 0j  # East
    cost[start, direction] = 0
    parents = {} # Used to reconstruct the path 
    visited = set()
    
    while True:
        print(f"{100*len(visited) / len(cost):.1f} %", end="\r")
        current, current_dir = min((node for node in cost if node not in visited), key=cost.get)
        visited.add((current, current_dir))
        
        if current == end:
            return cost[current, current_dir]
        
        for neighbour in graph[current]:
            delta_dir = neighbour - current
            new_cost = cost[current, current_dir] + move_cost(current, neighbour, current_dir)
            
            if new_cost < cost[neighbour, delta_dir]:
                cost[neighbour, delta_dir] = new_cost
                parents[neighbour, delta_dir] = [(current, current_dir)]
            elif new_cost == cost[neighbour, delta_dir]:
                # If the cost is the same, we can reach the node from multiple paths - so add the extra node
                parents[neighbour, delta_dir].append((current, current_dir))
    
    # Backtrack the parent pathto find all paths of the shortest length from start to end
    def backtrack(node):
        if node == start:
            return [[start]]
        paths = []
        for p in parents[node]:
            for path in backtrack(p):
                paths.append(path + [node])
        return paths
    import ipdb; ipdb.set_trace()
    paths = backtrack(end)


def dikstra_paths(graph: dict[complex, list[complex]], start: complex, end: complex) -> int:
    cost = {(node, _dir): float('inf') for _dir in [-1, 1, 1j, -1j] for node in graph}
    direction = 1 + 0j  # East
    cost[start, direction] = 0
    parents = {} # Used to reconstruct the path 
    visited = set()
    
    while len(visited) < len(cost):
        print(f"{100*len(visited) / len(cost):.1f} %", end="\r")
        current, current_dir = min((node for node in cost if node not in visited), key=cost.get)
        visited.add((current, current_dir))
        
        # if current == end:
        #     import ipdb; ipdb.set_trace()
        #     return cost[current, current_dir]
        
        for neighbour in graph[current]:
            delta_dir = neighbour - current
            new_cost = cost[current, current_dir] + move_cost(current, neighbour, current_dir)
            
            if new_cost < cost[neighbour, delta_dir]:
                cost[neighbour, delta_dir] = new_cost
                parents[neighbour, delta_dir] = [(current, current_dir)]
            elif new_cost == cost[neighbour, delta_dir]:
                # If the cost is the same, we can reach the node from multiple paths - so add the extra node
                parents[neighbour, delta_dir].append((current, current_dir))
    
    # Backtrack the parent pathto find all paths of the shortest length from start to end
    def backtrack(state) -> list[list[tuple[complex, complex]]]:
        node, direction = state
        if node == start:
            return [[(start, direction)]]
            
        paths = []
        if state in parents:
            for parent_state in parents[state]:
                for path in backtrack(parent_state):
                    paths.append(path + [state])
        return paths
    
    # Find the optimal end state(s)
    end_states = [(end, dir) for dir in [-1, 1, 1j, -1j]]
    min_cost = min(cost[state] for state in end_states)
    optimal_end_states = [state for state in end_states if cost[state] == min_cost]
    
    # Get all paths from each optimal end state
    all_paths = []
    for end_state in optimal_end_states:
        all_paths.extend(backtrack(end_state))

    total_nodes_visited = 0
    path_set = set()
    for path in all_paths:
        for node, _ in path:
            path_set.add(node)
    return len(path_set)


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
        graph, start, end = map_to_graph(input_data)
        return dikstra(graph, start, end)


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
        graph, start, end = map_to_graph(input_data)
        path_length = dikstra_paths(graph, start, end)
        # Now we find all the paths of the best length, and the full set of nodes that are part of any valid path them
        return path_length
        
        
