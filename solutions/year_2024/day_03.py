# solutions/year_2024/day_00.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer

"""
This one is all about parsing the string into the correct format, and extracting the logic block. 
Part1 is just mul(A,B) with no spaces, but the rest of the string looks like it's going to be more of the same. 

I know patterns like this use a stack to run through the string and do things like parenthesis matching. 
Or is there a tokenization method maybe? 

"""

def match_pattern(line, pattern):
    # Scan the line for the pattern, then fill a buffer with the contents of the pattern to check for matching brackets
    buffer = []
    has_match = False

    sum = 0
    for i in range(2, len(line)):
        match = line[i-3: i]
        if match == pattern:
            # Scan forwards to see if the brackets are correctly set
            has_match = True
            # print("Match found", match, i)
        
        if line[i] in "(" and has_match:
            # print("Open bracket found", i)
            open_idx = i
        
        elif line[i] in ")" and has_match:
            # print("Close bracket found", i)
            close_idx = i
            
            if "," in line[open_idx+1:close_idx]:
                a, b = line[open_idx+1:close_idx].split(",")
                sum += int(a)*int(b)
            has_match = False
        
        if has_match and line[i] not in "(,01234567890)":
            has_match = False
    return sum


def match_pattern_p2(lines, pattern):
    """
    Scans a line for a pattern and calculates sum of multiplications within brackets
    following the pattern, respecting do/don't controls.
    """
    buffer = []
    has_match = False
    do_enabled = True
    sum = 0
    open_idx = -1  # Initialize to invalid index
    
    # Need to start from 0 to catch all patterns
    for line in lines:
        for i in range(len(line)):
            # Check control statements - make sure to check complete words
            if i >= 4 and line[i-3:i+1] == "do()":
                do_enabled = True
                continue
                
            if i >= 7 and line[i-6:i+1] == "don't()":
                do_enabled = False
                continue
            
            # Only process if do is enabled
            if do_enabled:
                # Check for pattern match - need to ensure we have enough characters
                if i >= len(pattern)-1:
                    current_match = line[i-len(pattern)+1:i+1]
                    if current_match == pattern:
                        has_match = True
                        continue
                
                # Handle brackets
                if line[i] == '(' and has_match:
                    open_idx = i
                
                elif line[i] == ')' and has_match and open_idx != -1:
                    close_idx = i
                    
                    # Extract and process the content between brackets
                    bracket_content = line[open_idx+1:close_idx]
                    if ',' in bracket_content:
                        try:
                            a, b = bracket_content.split(',')
                            # Strip whitespace and convert to integers
                            sum += int(a.strip()) * int(b.strip())
                        except ValueError:
                            # Handle case where conversion to int fails
                            pass
                    
                    # Reset state after processing brackets
                    has_match = False
                    open_idx = -1
                
                # Reset match if we find invalid characters while looking for brackets
                elif has_match and line[i] not in "( ,0123456789)":
                    has_match = False
    
    return sum


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
        # Your solution for part 1 goes here
        run = 0
        for line in input_data:
            run += match_pattern(line, "mul")
        return run


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
        run = 0
        run += match_pattern_p2(input_data, "mul")
        return run
