# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer


def unpack_input(input_data: list[str]) -> list[any]:
    input_data = input_data[0]
    # Pairs of [file size, freespace] throughthe input
    output = []
    spaces = dict()
    file_identifier = 0
    for idx in range(len(input_data)):
        if idx % 2 == 0:
            size = int(input_data[idx])
            output.extend([idx // 2] * size)
        else:
            space = int(input_data[idx])
            spaces[len(output)] = space
            output.extend([None] * space)

    return output, spaces


def compress(memory_buffer: list[any]) -> list[any]:
    left = 0
    right = len(memory_buffer) - 1
    while left < right:
        if memory_buffer[left] is None:
            while memory_buffer[right] is None:
                right -= 1
            # Can move these pairs in a single line in python without creating intermediate buffers
            memory_buffer[left], memory_buffer[right] = (
                memory_buffer[right],
                memory_buffer[left],
            )
            left += 1
            right -= 1
        else:
            left += 1
    return memory_buffer


def compute_checksum(memory_buffer: list[any]) -> int:
    checksum = 0
    for idx in range(len(memory_buffer)):
        if memory_buffer[idx] is not None:
            checksum += memory_buffer[idx] * idx  # Zero indexed
    return checksum


def compress_whole_files(
    memory_buffer: list[any], spaces: dict, num_files: int
) -> list[any]:
    # Here we nee to find the whole blocks of memory where the entire file can be moved to

    left = 0
    # print(memory_buffer, len(memory_buffer))
    right = len(memory_buffer) - 1
    for _ in range(num_files):
        left = -1

        # Start by finding the last file - and it's size
        while memory_buffer[right] is None:
            right -= 1
        last_file = memory_buffer[right]
        last_file_size = 0
        while memory_buffer[right] == last_file:
            last_file_size += 1
            right -= 1

        # Then we find the leftmost space large enough to hold the file
        # print(spaces)
        keys = sorted(list(spaces.keys()))
        # print(keys)
        for space in keys:
            # print(space, spaces[space], last_file_size)
            if space < right:
                if spaces[space] >= last_file_size:
                    left = space
                    break

        if left > 0:
            prev_len = len(memory_buffer)
            # print(left, right, last_file_size)
            memory_buffer[left : left + last_file_size] = [last_file] * last_file_size
            assert len(memory_buffer) == prev_len
            memory_buffer[right + 1 : right + 1 + last_file_size] = [
                None
            ] * last_file_size
            assert len(memory_buffer) == prev_len
            # Update the spaces dictionary
            # print(left, right, space)
            if spaces[left] > last_file_size:
                # print(left, last_file_size)
                spaces[left + last_file_size] = spaces.pop(left) - last_file_size
            else:
                spaces.pop(left)

        # print(memory_buffer, len(memory_buffer), _)
    return memory_buffer


def part1(input_data: list[str] | None) -> str | int:
    """
    Solve part 1 of the day's challenge.

    This problem takes a string representation of file blocks, and spaces between as arranged in contious memory.
    Then the task is to compress this into a single block of memory. (Defragmentation)

    Suspect this will be a two pointer kind of problem, as we are told to take mmory from the end and move to the earliest free space.


    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 1"):
        memory_buffer, _ = unpack_input(input_data)
        defrag = compress(memory_buffer)
        checksum = compute_checksum(defrag)
        return checksum


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
        memory_buffer, spaces = unpack_input(input_data)
        num_files = len(set(memory_buffer)) - 1
        defrag = compress_whole_files(memory_buffer, spaces, num_files=num_files)
        checksum = compute_checksum(defrag)
        return checksum
