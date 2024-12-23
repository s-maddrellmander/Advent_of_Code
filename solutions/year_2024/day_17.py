# solutions/year_2024/day_17.py

from logger_config import logger
from utils import Timer


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
        infile = "inputs/year_2024/day_17.txt"
        ans = 0
        D = open(infile).read().strip()

        regs, program = D.split("\n\n")
        A, B, C = [int(x.split(":")[1].strip()) for x in regs.split("\n")]
        program = program.split(":")[1].strip().split(",")
        program = [int(x) for x in program]

        def getCombo(x):
            if x in [0, 1, 2, 3]:
                return x
            if x == 4:
                return A
            if x == 5:
                return B
            if x == 6:
                return C
            return -1

        A = A
        B = 0
        C = 0
        ip = 0
        out = []
        while True:
            if ip >= len(program):
                out = "".join([str(x) for x in out])
                return out
            cmd = program[ip]
            op = program[ip + 1]
            combo = getCombo(op)

            # print(ip, len(program), cmd)
            if cmd == 0:
                A = A // 2**combo
                ip += 2
            elif cmd == 1:
                B = B ^ op
                ip += 2
            elif cmd == 2:
                B = combo % 8
                ip += 2
            elif cmd == 3:
                if A != 0:
                    ip = op
                else:
                    ip += 2
            elif cmd == 4:
                B = B ^ C
                ip += 2
            elif cmd == 5:
                out.append(int(combo % 8))
                ip += 2
            elif cmd == 6:
                B = A // 2**combo
                ip += 2
            elif cmd == 7:
                C = A // 2**combo
                ip += 2


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
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
