import pytest

from solutions.year_2023.day_05 import *

DATA = [
    "seeds: 79 14 55 13",
    "",
    "seed-to-soil map:",
    "50 98 2",
    "52 50 48",
    "",
    "soil-to-fertilizer map:",
    "0 15 37",
    "37 52 2",
    "39 0 15",
    "",
    "fertilizer-to-water map:",
    "49 53 8",
    "0 11 42",
    "42 0 7",
    "57 7 4",
    "",
    "water-to-light map:",
    "88 18 7",
    "18 25 70",
    "",
    "light-to-temperature map:",
    "45 77 23",
    "81 45 19",
    "68 64 13",
    "",
    "temperature-to-humidity map:",
    "0 69 1",
    "1 0 69",
    "",
    "humidity-to-location map:",
    "60 56 37",
    "56 93 4",
]


def test_find_element_in_list():
    assert find_element_in_list("a", ["a", "b", "c"]) == 0
    assert find_element_in_list("b", ["a", "b", "c"]) == 1
    assert find_element_in_list("d", ["a", "b", "c"]) == -1


@pytest.mark.skip("Method changed")
def test_parse_data():
    seeds, result = parse_data(DATA)

    assert seeds == {79, 14, 55, 13}
    assert len(result["seed-to-soil map:"]) == 50
    assert len(result["soil-to-fertilizer map:"]) == 54
    assert len(result["fertilizer-to-water map:"]) == 61
    assert len(result["water-to-light map:"]) == 77
    assert len(result["light-to-temperature map:"]) == 55
    assert len(result["temperature-to-humidity map:"]) == 70
    assert len(result["humidity-to-location map:"]) == 41


def test_get_location():
    seeds, mapping = parse_data(DATA)
    assert get_location_for_seed(79, mapping) == 82
    assert get_location_for_seed(14, mapping) == 43
    assert get_location_for_seed(55, mapping) == 86
    assert get_location_for_seed(13, mapping) == 35


def test_mapping_stages():
    seeds, mapping = parse_data(DATA)

    soil = map_func(79, mapping["seed-to-soil map:"])
    assert soil == 81
    fertilizer = map_func(soil, mapping["soil-to-fertilizer map:"])
    assert fertilizer == 81
    water = map_func(fertilizer, mapping["fertilizer-to-water map:"])
    assert water == 81
    light = map_func(water, mapping["water-to-light map:"])
    assert light == 74
    temperature = map_func(light, mapping["light-to-temperature map:"])
    assert temperature == 78
    humidity = map_func(temperature, mapping["temperature-to-humidity map:"])
    assert humidity == 78
    location = map_func(humidity, mapping["humidity-to-location map:"])
    assert location == 82


def test_part1():
    assert part1(DATA) == 35


def test_part2():
    assert part2(DATA) == 46


def test_part2_single_val():
    seeds, mapping = parse_data(DATA)
    # Test the single value
    assert get_location_for_seed(82, mapping=mapping) == 46


def test_investigate_part_2():
    seeds, mapping = parse_data(DATA)

    tmp = 0
    for seed in range(0, 100):
        loc = get_location_for_seed(seed, mapping=mapping)
        logger.debug(f"Location for seed {seed} is {loc}")


def test_map_func():
    map_dict = {1: (2, 3), 4: (5, 6), 10: (8, 9)}

    assert map_func(0, map_dict) == 0
    # In the first mapper
    assert map_func(1, map_dict) == 2
    assert map_func(2, map_dict) == 3
    assert map_func(3, map_dict) == 4
    # In the second mapper
    assert map_func(4, map_dict) == 5
    assert map_func(5, map_dict) == 6
    assert map_func(6, map_dict) == 7
    assert map_func(7, map_dict) == 8
    assert map_func(8, map_dict) == 9
    assert map_func(9, map_dict) == 10
    # In the third mapper
    assert map_func(10, map_dict) == 8
    # Test the top end - should return the same value
    assert map_func(19, map_dict) == 19


#    4917124
