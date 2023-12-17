# solutions/year_2023/day_15.py
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def parser(input_data: List[str]) -> Dict[Tuple[int, int], str]:
    coords: dict = {}
    for i, line in enumerate(input_data):
        line = line.strip()
        for j, loc in enumerate(line):
            # if loc != ".":
            coords[(int(j), int(i))] = loc
    return coords


def print_grid(coordinates):
    if not coordinates:
        print("No coordinates provided.")
        return

    # Determine the bounds of the grid
    max_col = max(coordinates, key=lambda x: x[0])[0]
    max_row = max(coordinates, key=lambda x: x[1])[1]

    # Iterate over each row and column, printing the appropriate character
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            if (col, row) in coordinates:
                print("#", end="")
            else:
                print(".", end="")
        print()  # New line at the end of each row


def fn(current, next_state, prev, state_dir_pairs, queue):
    if (current, next_state) in state_dir_pairs:
        return current, next_state, prev, state_dir_pairs, queue
    else:
        # Add the state, dir pair to the set
        state_dir_pairs.add((current, next_state))
        queue.append((next_state, current))  # becomes (current, prev)
        return current, next_state, prev, state_dir_pairs, queue


def ray_tracing(
    coords: Dict[Tuple[int, int], str],
    start: Tuple[int, int],
    prev: Optional[Tuple[int, int] | None] = None,
) -> List[Tuple[int, int]]:
    """
    Rays move in straight line
    | is a splitter, if the ray hits the splitter from the side it splits into two rays, going up and down
    - is a splitter, if the ray hits the splitter from the top or bottom it splits into two rays, going left and right
    / is a mirror that reflects the ray 90 degrees
    \ is a mirror that reflects the ray 90 degrees
    """
    path: List = list()
    if prev is None:
        prev = (start[0] - 1, start[1])  # Moving to the right
    queue = deque()  # type: deque
    queue.append((start, prev))

    state_dir_pairs: set = set()

    while queue:
        # logger.debug(f"Queue: {len(queue)}, {len(set(path))}")
        # logger.debug(f"Queue: {len(queue)}")

        # print_grid(path)
        current, prev = queue.popleft()
        x, y = current
        if current not in coords:
            continue
        else:
            path.append(current)
            if coords[current] == "|":
                # if previous state is up just continue through
                if prev == (x, y - 1):
                    next_state = (x, y + 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )
                elif prev == (x, y + 1):
                    next_state = (x, y - 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )
                else:  # If from either side, split
                    next_state = (x, y + 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )
                    next_state = (x, y - 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

            elif coords[current] == "-":
                if prev == (x - 1, y):
                    next_state = (x + 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x + 1, y):
                    next_state = (x - 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                else:
                    next_state = (x + 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                    next_state = (x - 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

            elif coords[current] == "/":
                if prev == (x - 1, y):
                    next_state = (x, y - 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x + 1, y):
                    next_state = (x, y + 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x, y - 1):
                    next_state = (x - 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x, y + 1):
                    next_state = (x + 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

            elif coords[current] == "\\":
                if prev == (x - 1, y):
                    next_state = (x, y + 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x + 1, y):
                    next_state = (x, y - 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x, y - 1):
                    next_state = (x + 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x, y + 1):
                    next_state = (x - 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

            elif coords[current] == ".":
                # Continue in the same direction
                if prev == (x, y - 1):
                    next_state = (x, y + 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x, y + 1):
                    next_state = (x, y - 1)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x - 1, y):
                    next_state = (x + 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

                elif prev == (x + 1, y):
                    next_state = (x - 1, y)
                    current, next_state, prev, state_dir_pairs, queue = fn(
                        current, next_state, prev, state_dir_pairs, queue
                    )

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
        mirrors = parser(input_data)
        return len(set(ray_tracing(mirrors, (0, 0))))


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

    energies = []
    with Timer("Part 2"):
        mirrors = parser(input_data)
        rows = len(input_data)
        cols = len(input_data[0])
        top_row = [(col, 0) for col in range(cols)]
        prevs_top = [(col, -1) for col in range(cols)]

        bottom_row = [(col, rows - 1) for col in range(cols)]
        prevs_bottom = [(col, rows) for col in range(cols)]

        left_side = [(0, row) for row in range(rows)]
        left_prevs = [(-1, row) for row in range(rows)]

        right_side = [(cols - 1, row) for row in range(rows)]
        right_prevs = [(cols, row) for row in range(rows)]

        starts = top_row + right_side + bottom_row[::-1] + left_side[::-1]
        prevs = prevs_top + right_prevs + prevs_bottom[::-1] + left_prevs[::-1]
        for start, prev in zip(starts, prevs):
            energies.append(len(set(ray_tracing(mirrors, start, prev))))
        return max(energies)
