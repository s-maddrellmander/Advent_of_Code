# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer
import heapq


def parse_map(input_data, linit_limit=1024):
    carte = {}
    for line in input_data[:linit_limit]:
        carte[complex(*map(int, line.split(',')))] = '#'
    return carte    

def print_map(carte, path=None, limit=7):
    for y in range(limit):
        for x in range(limit):
            if complex(x, y) in carte:
                print('#', end='')
            elif path and complex(x, y) in path:
                print('o', end='')
            else:
                print('.', end='')
        print()

def a_star(carte, start, end, map_limit=71):
    # Let's try using A* path finding algorithm  
    frontier = []
    counter = 0
    heapq.heappush(frontier, (0, counter, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    
    while frontier:
        current = heapq.heappop(frontier)[-1]        
        if current == end: 
            break
        
        for dir in [1, 1j, -1, -1j]:
            # Check in the 4x dirs, only blocks are on the map, so we need limits too
            if current + dir not in carte and (current+dir).imag >= 0 and (current+dir).real >= 0 and (current+dir).imag < map_limit and (current+dir).real < map_limit:
                new_cost = cost_so_far[current] + 1 # Only costs 1 to move to a new block
                new_loc = current + dir
                if new_loc not in cost_so_far or new_cost < cost_so_far[new_loc]:
                    cost_so_far[new_loc] = new_cost
                    priority = new_cost + abs(end - new_loc) # Manhattan distance
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, new_loc))
                    came_from[new_loc] = current
                    
    return came_from, cost_so_far

def reconstruct_path(came_from, start, end):    
    if end not in came_from:
        return None
    current = end
    
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
                
    

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
        carte = parse_map(input_data, 1024)
        start = complex(0, 0)
        end = complex(70, 70)
        came_from, cost_so_far = a_star(carte, start, end, 71)
        path = reconstruct_path(came_from, start, end)
        # print_map(carte, path, limit=71)
        return len(path) - 1


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
        # Find the the coord of the first block that stops the path - binary search
        left = 0
        right = len(input_data)
        while left < right:
            mid = (left + right) // 2
            carte = parse_map(input_data, mid)
            start = complex(0, 0)
            end = complex(70, 70)
            came_from, cost_so_far = a_star(carte, start, end, 71)
            path = reconstruct_path(came_from, start, end)
            if path:
                left = mid + 1
            else:
                right = mid
        
        return input_data[left-1]
        
        
