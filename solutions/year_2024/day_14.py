# solutions/year_2024/day_14.py

from logger_config import logger
from utils import Timer


def parse_data(lines: list[str]) -> list[tuple[complex, complex]]:
    guards = []
    for line in lines:
        line = line.split(" ")
        guard_pos = line[0].split("=")[1].split(",")
        guard_pos = complex(int(guard_pos[0]), int(guard_pos[1]))

        guard_vel = line[1].split("=")[1].split(",")
        guard_vel = complex(int(guard_vel[0]), int(guard_vel[1]))
        guards.append((guard_pos, guard_vel))
    return guards


def move_guard(
    guard: tuple[complex, complex], t: int, bounds: tuple[int, int]
) -> tuple[int, int]:
    pos, vel = guard
    pos = pos + vel * t
    x = int(pos.real) % bounds[0]
    y = int(pos.imag) % bounds[1]
    return x, y


def put_in_quadrant(guard: tuple[int, int], bounds: tuple[int, int]) -> int:
    mid_x = bounds[0] // 2
    mid_y = bounds[1] // 2
    # print(f"Mid x: {mid_x}, Mid y: {mid_y}")
    if guard[0] < mid_x and guard[1] < mid_y:
        return 0
    elif guard[0] > mid_x and guard[1] < mid_y:
        return 1
    elif guard[0] > mid_x and guard[1] > mid_y:
        return 2
    elif guard[0] < mid_x and guard[1] > mid_y:
        return 3
    else:
        return -1  # These are on the mid lines


def part1(
    input_data: list[str] | None, bounds: tuple[int, int] = (101, 103)
) -> str | int:
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
        guards = parse_data(input_data)
        t = 100
        quads = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
        for guard in guards:
            x, y = move_guard(guard, t, bounds)  # 0 index, grid coords
            quad = put_in_quadrant((x, y), bounds)
            quads[quad] += 1
        sol = quads[0] * quads[1] * quads[2] * quads[3]
        return sol


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
        # Find the christmas tree picture

        var_xs = []
        var_ys = []

        for t in range(0, 10000):
            print(f"{t}/10000", end="\r")
            _guards = parse_data(input_data)
            guards = [move_guard(guard, t, (101, 103)) for guard in _guards]

            mean_x = sum([guard[0] for guard in guards]) // len(guards)
            mean_y = sum([guard[1] for guard in guards]) // len(guards)
            var_x = sum([(guard[0] - mean_x) ** 2 for guard in guards]) // len(guards)
            var_y = sum([(guard[1] - mean_y) ** 2 for guard in guards]) // len(guards)
            var_xs.append(var_x)
            var_ys.append(var_y)

            if var_x < 350 and var_y < 350:
                return t

        #     print(f"t: {t}, mean_x: {mean_x}, mean_y: {mean_y}, var_x: {var_x}, var_y: {var_y}")
        # # Find the index of the minimum var x and var y (seperate)
        # min_var_x = min(var_xs)
        # idx_x = var_xs.index(min_var_x)
        # min_var_y = min(var_ys)
        # idx_y = var_ys.index(min_var_y)

        # # What step when these coencide?

        # return min(idx_x, idx_y)
