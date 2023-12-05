# solutions/year_2023/day_00.py
from typing import List, Optional, Union, Dict

from logger_config import logger
from utils import Timer
from multiprocessing import Pool
from functools import partial


from tqdm import tqdm


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

def find_largest_smaller_than_x(numbers, x):
    # Filter the list to only include numbers smaller than or equal to x
    smaller_numbers = [num for num in numbers if num <= x]

    # If there are no numbers smaller than or equal to x, return x
    if not smaller_numbers:
        return min(numbers)

    # Return the maximum number from the filtered list
    return max(smaller_numbers)

 
def parse_data(data: List[str]) -> Union[list[int], dict]:
    maps_set: Dict[str, Dict[int, int]] = {
        "seed-to-soil map:": {},
        "soil-to-fertilizer map:": {},
        "fertilizer-to-water map:" : {},
        "water-to-light map:" : {},
        "light-to-temperature map:": {} ,
        "temperature-to-humidity map:": {},
        "humidity-to-location map:": {}
    }
    # Parse the input data into a set of maps
    seeds = data[0]
    seeds = seeds.split(":")[1].strip()
    seeds = [int(x) for x in seeds.split(" ")]
    
    for row in tqdm(data[1:]):
        if row == "":
            continue
        if row in maps_set.keys():
            current_map = row
            continue
        else:
            # parse the row
            try:
                row = [int(x) for x in row.split(" ")]
                
                # Use a range based approach to map the values
                # ((start, end), (start, end), range)
                # This gives ((range for source), (range for destination), range)
                maps_set[current_map][row[1]] = (row[0], row[2])
            except ValueError:
                print("Error: Could not convert string to integer.")
            except IndexError:
                print("Error: Not enough elements in the row.")
    return seeds, maps_set
            

def map_func(x: int, map_dict: dict) -> int:
    # Map the value to the next value - if not found return the value
    phi = find_largest_smaller_than_x(list(map_dict.keys()), x)
    if x < phi:
        return x
    else:
        delta = x - phi
        if delta >= map_dict[phi][1]:
            return x
        y_hat = map_dict[phi][0] + delta
        return y_hat
        


def get_location_for_seed(seed: int, mapping: dict) -> int:
    soil = map_func(seed, mapping["seed-to-soil map:"])
    fertilizer = map_func(soil, mapping["soil-to-fertilizer map:"])
    water = map_func(fertilizer, mapping["fertilizer-to-water map:"])
    light = map_func(water, mapping["water-to-light map:"])
    temperature = map_func(light, mapping["light-to-temperature map:"])
    humidity = map_func(temperature, mapping["temperature-to-humidity map:"])
    location = map_func(humidity, mapping["humidity-to-location map:"])
    return location

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
        seeds, mapping = parse_data(input_data)
        locations = []
        for seed in seeds:
            location = get_location_for_seed(seed, mapping)
            logger.debug(f"Location for seed {seed} is {location}")
            locations.append(location)   
        return min(locations)


def coarse(seeds, mapping, coarse_step, min_seed, min_value):
    # Coarse search
    tmp_strt = None
    tmp_end = None
    for i in range(0, len(seeds), 2):
        for seed in range(seeds[i], seeds[i]+ seeds[i+1], coarse_step):
            value = get_location_for_seed(seed, mapping)
            if value < min_value:
                min_value = value
                min_seed = seed
                tmp_strt = seeds[i]
                tmp_end =  seeds[i]+ seeds[i+1]   
                logger.debug(f"Coarse search: min_seed: {min_seed}, min_value: {min_value} Start {tmp_strt} End {tmp_end}" ) 
    return min_seed, min_value, tmp_strt, tmp_end
        
def fine_search(seeds, mapping, min_seed, min_value, fine_steps, min_range, max_range):
    for fine_step in fine_steps:
        start = max(min_seed - fine_step * 10, min_range)
        end = min(min_seed + fine_step * 10, max_range)
        # seed_index = seeds.index(min_seed)
        # end = end, seeds[seed_index]+seeds[seed_index+1])

        logger.debug(f"Fine search: start: {start}, end: {end}, fine_step: {fine_step}, min_seed: {min_seed}, min_value: {min_value}")
        for seed in range(start, end, fine_step):
            value = get_location_for_seed(seed, mapping)
            if value < min_value:
                min_value = value
                min_seed = seed
    return min_seed, min_value

def coarse_to_fine_search(seeds, mapping, coarse_step, fine_steps):
    min_value = float('inf')
    min_seed = None

    min_seed, min_value, start, end = coarse(seeds, mapping, coarse_step, min_seed, min_value)
    logger.debug(f"Coarse search: min_seed: {min_seed}, min_value: {min_value}")
    
    # min and max range are the seed limits the min_seed lies in
    min_range = start # seeds[find_largest_smaller_than_x(min_seed, seeds)]
    max_range = end # float("inf") # min_range + seeds[find_largest_smaller_than_x(min_seed, seeds) + 1]
    assert min_range <= min_seed <= max_range
    logger.debug(f"min_range: {min_range}, max_range: {max_range}")
    
    # Fine searches - assuming the min value is in the range for a sinlge seed pair
    min_seed, min_value = fine_search(seeds, mapping, min_seed, min_value, fine_steps, min_range, max_range)

    return min_value

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
        seeds, mapping = parse_data(input_data)
        # 1. Scan each pair of seeds range in a coarse 1000000 step size
        # 2. Find the smallest value from all steps
        # 3. Scan the range in a fine 10000 over +/- 1000000 around the smallest value
        # 4. Find the smallest value from all steps
        # 5. Scan the range in a fine 100 over +/- 10000 around the smallest value
        # 6. Find the smallest value from all steps
        # 7. Scan the range in a fine 1 over +/- 1 around the smallest value
        
        min_seed = coarse_to_fine_search(seeds, mapping, 100000, [100, 10, 1])
        return min_seed
