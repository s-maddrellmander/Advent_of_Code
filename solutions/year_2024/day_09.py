# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer

def unpack_input(input_data: list[str]) -> list[any]:
    input_data = input_data[0]
    # Pairs of [file size, freespace] throughthe input
    output = []
    file_identifier = 0
    for idx in range(len(input_data)):
        if idx % 2 == 0:
            size = int(input_data[idx])
            output.extend([idx//2] * size)
        else:
            space = int(input_data[idx])
            output.extend([None] * space)
        
    return output


def compress(memory_buffer: list[any]) -> list[any]:
    left = 0
    right = len(memory_buffer) - 1
    while left < right:
        if memory_buffer[left] is None:
            while memory_buffer[right] is None:
                right -= 1
            # Can move these pairs in a single line in python without creating intermediate buffers
            memory_buffer[left], memory_buffer[right] = memory_buffer[right], memory_buffer[left]
            left += 1
            right -= 1
        else:
            left += 1
    return memory_buffer

def compute_checksum(memory_buffer: list[any]) -> int:
    checksum = 0
    for idx in range(len(memory_buffer)):
        if memory_buffer[idx] is not None:
            checksum += memory_buffer[idx] * idx # Zero indexed
    return checksum

def compress_whole_files(memory_buffer: list[any]) -> list[any]:
    pass


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
        memory_buffer = unpack_input(input_data)
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
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
