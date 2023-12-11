# solutions/year_2023/day_10.py
from collections import deque
from typing import List, Optional, Union

from logger_config import logger
from utils import Timer

# def valid_direction(current, next_dir):
#     """ Check if the movement from the current pipe to the next direction is valid """
#     if current == '|' and next_dir in ['|', 'L', 'J']:
#         return True
#     elif current == '-' and next_dir in ['-', '7', 'F']:
#         return True
#     elif current in ['L', 'J', '7', 'F'] and next_dir in ['|', '-', 'L', 'J', '7', 'F']:
#         return True
#     return False

# # Function to determine the correct pipe for the 'S' position
# def determine_correct_pipe_for_s(grid, s_pos):
#     x, y = s_pos
#     neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # North, South, West, East
#     neighbor_pipes = {grid[nx][ny] for nx, ny in neighbors if 0 <= nx < len(grid) and 0 <= ny < len(grid[0])}

#     # Pipe types and their compatible directions
#     pipe_types = {
#         'F': {'|', '-','.'},
#         # '-': {'-', '7', 'F','.'},
#         # 'L': {'|', '-', 'L', 'J', '7', 'F','.'},
#         # 'J': {'|', '-', 'L', 'J', '7', 'F','.'},
#         # '7': {'|', '-', 'L', 'J', '7', 'F','.'},
#         # 'F': {'|', '-', 'L', 'J', '7', 'F','.'}
#     }

#     for pipe, compatible in pipe_types.items():
#         if neighbor_pipes.issubset(compatible):
#             return pipe

#     return None  # If no compatible pipe is found


# # Function to parse the grid and find the starting position
# def parse_grid(grid):
#     parsed_grid = [list(line.strip()) for line in grid]
#     start_pos = None
#     for i, row in enumerate(parsed_grid):
#         for j, cell in enumerate(row):
#             if cell == 'S':
#                 start_pos = (i, j)
#     return parsed_grid, start_pos

# # Function to find all starting positions 'S' in the grid
# def find_start_positions(grid):
#     start_positions = []
#     for i, row in enumerate(grid):
#         for j, cell in enumerate(row):
#             if cell == 'S':
#                 start_positions.append((i, j))
#     return start_positions

# # Function to inspect surrounding tiles of a position
# def inspect_surroundings(grid, position):
#     x, y = position
#     surroundings = {}
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # north, south, west, east
#     for dx, dy in directions:
#         nx, ny = x + dx, y + dy
#         if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
#             surroundings[(nx, ny)] = grid[nx][ny]
#     return surroundings

# def bfs_full_paths(grid, start_pos):
#     # Directions: north, south, east, west
#     # TODO: Use this to define the grid {(x, y)" [types of direction that would be valid]]}"
#     # THen check if those coordinates + the current ones are in the valid set etc.
#     dir_moves = {'|': [(-1, 0), (1, 0)], '-': [(0, 1), (0, -1)],
#                  'L': [(1, 0), (0, 1)], 'J': [(1, 0), (0, -1)],
#                  '7': [(-1, 0), (0, -1)], 'F': [(-1, 0), (0, 1)],
#                  'S': [(-1, 0), (1, 0), (0, -1), (0, 1)]}  # 'S' can connect in any direction

#     if not start_pos:
#         return "Starting position not found"

#     rows, cols = len(grid), len(grid[0])
#     visited = set()
#     queue = deque([(start_pos, [start_pos])])  # Queue now holds tuples of position and path to that position

#     all_paths = []

#     while queue:
#         (x, y), path = queue.popleft()
#         if (x, y) not in visited:
#             visited.add((x, y))

#             current_pipe = grid[x][y]
#             for dx, dy in dir_moves.get(current_pipe, []):
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
#                     next_pipe = grid[nx][ny]
#                     if next_pipe != '.' and valid_direction(current_pipe, next_pipe):
#                         new_path = path + [(nx, ny)]  # Extend the current path with the new position
#                         queue.append(((nx, ny), new_path))
#                         if next_pipe in ['|', '-', 'L', 'J', '7', 'F']:  # If next pipe is a valid pipe, save the path
#                             all_paths.append(new_path)

#     return all_paths

# ROW, COL
direction_types = {
    "|": [(-1, 0), (1, 0)],
    "-": [(0, 1), (0, -1)],
    "L": [(-1, 0), (0, 1)],
    "J": [(-1, 0), (0, -1)],
    "7": [(1, 0), (0, -1)],
    "F": [(1, 0), (0, 1)],
    "S": [(-1, 0), (0, 1)],
}
# valid_connections = {
#     "|": ["|", "L", "J", "7", "F"],
#     "-": ["-", "L", "J", "7", "F"],
#     "L": ["|", "-", "L", "J", "7", "F"],
#     "7": ["|", "-", "L", "J", "7", "F"],
#     "J": ["|", "-", "L", "J", "7", "F"],
#     "F": ["|", "-", "L", "J", "7", "F"],
# }


def parse_grid(grid: List[str]) -> dict:
    """
    Function to parse the grid and find the starting position
    """
    parsed_grid = [list(line.strip()) for line in grid]
    start_pos = None
    grid = {}
    for i, row in enumerate(parsed_grid):
        for j, cell in enumerate(row):
            if cell == "S":
                start_pos = (i, j)
            if cell != ".":
                valid_dirs = direction_types[cell]
                # if i == 3 and j == 3:
                #     import ipdb; ipdb.set_trace()
                # import ipdb; ipdb.set_trace()
                if any(
                    i + xy[0] >= 0
                    and j + xy[1] >= 0
                    and i + xy[0] < len(parsed_grid)
                    and j + xy[1] < len(parsed_grid[0])
                    and parsed_grid[i + xy[0]][j + xy[1]] != "."
                    and tuple([-1 * x for x in xy])
                    in direction_types.get(parsed_grid[i + xy[0]][j + xy[1]], [])
                    for xy in valid_dirs
                ):
                    logger.debug(f"{i}, {j} cell {cell} is valid")
                    for xy in valid_dirs:
                        temp = []
                        if i + xy[0] < len(parsed_grid) and j + xy[1] < len(
                            parsed_grid[0]
                        ):
                            if i + xy[0] >= 0 and j + xy[1] >= 0:
                                if parsed_grid[i + xy[0]][j + xy[1]] != ".":
                                    temp.append(xy)
                        if len(temp) == len(valid_dirs):
                            if (i, j) not in grid:
                                grid[(i, j)] = []
                            grid[(i, j)].extend(temp)
                            logger.debug(f"Adding {temp} to {i}, {j}")

    return grid, start_pos


# For y in lines:
# For c in row:
# If lines[y][x] not in grid and not “.”:
# Grid[{y, x}] = [ xy for xy in valid direction for type if {y, x} + xy in valid connection for type]
# #gives us a grid duct with valid connections


# Queue = [start]
# Visited []

# # we assume the path has a single loop and no branches
# While queue:
# Current = queue.pop()
# If current in visited or is S:
# Return
# Else:
# Visited.append(current)
# Queue.append(current.directions)
# # one of these will be where we came from so will be skipped later

# Return len(visited)// 2. # check for an off by one error here.


def plot_map(coordinates):
    if not coordinates:
        return "No coordinates provided."

    # Determine the bounds of the grid
    max_x = max(coordinates, key=lambda x: x[0])[0]
    max_y = max(coordinates, key=lambda x: x[1])[1]

    # Create and display the grid
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            if (x, y) in coordinates:
                print("#", end="")
            else:
                print(".", end="")
        print()  # New line at the end of each row


def bfs(grid: dict, start_pos: tuple) -> list:
    """
    Given the starting position, perform BFS on the pipe to find the path
    that returns to the starting position.
    start_pos: tuple of (x, y) coordinates (row, col)

    """
    logger.debug(f"{grid.keys()}")
    queue = deque([start_pos])
    visited = set()
    path = []
    while queue:
        y, x = queue.popleft()
        logger.debug(f"Visiting {y}, {x}")
        logger.debug(f"Queue: {queue}")
        logger.debug(f"dirs {grid[(y, x)]}")
        if (y, x) not in visited:
            visited.add((y, x))
            path.append((y, x))
            for dy, dx in grid[(y, x)]:
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited:
                    logger.debug(f"Adding {ny}, {nx} to queue")
                    queue.append((ny, nx))
    return path


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
        # Parsing the grid and finding the path
        parsed_grid, start_pos = parse_grid(input_data)
        # Finding the correct pipe for the 'S' position
        # s_position = find_start_positions(parsed_grid)[0]
        # correct_pipe = determine_correct_pipe_for_s(parsed_grid, start_pos)
        # parsed_grid[start_pos[0]][start_pos[1]] = correct_pipe

        path = bfs(parsed_grid, start_pos)

        # Displaying the result
        logger.info(parsed_grid)
        logger.info(f"Path: {path}")
        return "Part 1 solution not implemented."


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
