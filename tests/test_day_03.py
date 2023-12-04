import pytest
from solutions.year_2023.day_03 import *

@pytest.mark.skip(reason="Not implemented")
def test_process_array():
    # Test the foratting on the input array -> np array works as expected
    small_examples = ["467..114..",
                    "...*...&..",
                    "..35..633."]
    output = process_array(small_examples)
    assert output.shape == (3, 10)
    assert output[0, 0] == 4
    assert output[0, 1] == 6
    assert output[1, 3] == -1
    assert output[1, 4] == -99
    assert output[1, 7] == -9
    
    # Test the wildcards are correctly mapped
    wild = ["*@#$+-=%&/."]
    output = process_array(wild)
    assert np.allclose(output, np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 0]))
    
def test_get_wildcards():
    grid = np.array([[1, -1, 2], [-2, 3, -3], [4, -4, 5]])
    expected_output = (np.array([0, 1, 1, 2]), np.array([1, 0, 2, 1]))
    assert np.array_equal(get_wildcards(grid), expected_output)

def test_get_numbers():
    grid = np.array([[1, -1, 2], [-2, 3, -3], [4, -4, 5]])
    expected_output = (np.array([0, 0, 1, 2, 2]), np.array([0, 2, 1, 0, 2]))
    assert np.array_equal(get_numbers(grid), expected_output)
    
def test_find_adjacent_numbers():
    wildcards = np.array([[0, 1], [1, 0]])
    numbers = np.array([[0, 0, 1, 2, 2], [0, 2, 1, 0, 2]])
    expected_output = np.array([[0, 0, 1, 2], [0, 2, 1, 0]])
    assert np.array_equal(find_adjacent_numbers(wildcards, numbers), expected_output)
    
def test_find_adjacent_numbers_no_adjacent():
    # Single wildcard - top corner
    wildcards = np.array([[0], [0]])
    # Numbers not adjacent
    numbers = np.array([[2, 2, 3], [1, 2, 3]])
    expected_output = np.array([[], []])
    logger.info(find_adjacent_numbers(wildcards, numbers))
    assert np.array_equal(find_adjacent_numbers(wildcards, numbers), expected_output)
    
def test_find_adjacent_single_simple():
    # Single wildcard - top corner
    wildcards = np.array([[0], [0]])
    # Numbers adjacent
    numbers = np.array([[1], [1]])
    # Theses are the coordinates of the adjacent numbers
    expected_output = np.array([[1], [1]])
    assert np.array_equal(find_adjacent_numbers(wildcards, numbers), expected_output)

def test_find_adjacent_unique():
    # Example usage
    wildcards = np.array([[0, 1], [1, 0]])
    numbers = np.array([[0, 0, 1, 2, 2], [0, 2, 1, 0, 2]])  # Remember these are the coordinates
    adjacent_coords = find_adjacent_numbers(wildcards, numbers)
    assert np.array_equal(adjacent_coords, np.array([[0, 0, 1, 2], [0, 2, 1, 0]]))    

def test_get_number_sequences():
    grid = np.array([[1, 1, -1, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    numbers = np.array([[0, 1, 3], [0, 0, 2]])
    is_adjacent = np.array([[1], [0]])
    expected_output = [11]
    assert get_number_sequences(grid, numbers, is_adjacent) == expected_output
    
def test_get_number_sequences_multiple():
    grid = np.array([[1, 1, -1, 0], [0, 0, 0, 0], [1, -1, 1, 1]])
    numbers = np.array([[0, 1, 0, 2, 3], [0, 0, 2, 2, 2]])
    is_adjacent = np.array([[1, 0, 2], [0, 2, 2]])
    expected_output = [11, 1, 11]
    assert get_number_sequences(grid, numbers, is_adjacent) == expected_output

@pytest.mark.skip(reason="Not implemented")
def test_part1():
    test_input = [
            "467..114..",
            "...*......",
            "..35..633.",
            "......#...",
            "617*......",
            ".....+.58.",
            "..592.....",
            "......755.",
            "...$.*....",
            ".664.598.."
    ]
    grid = process_array(test_input)
    logger.info(grid)
    wildcards = tuple_array_to_array(get_wildcards(grid))
    logger.info(wildcards)
    assert wildcards.shape == (2, 6)
    assert np.all(wildcards[:, :3] == np.array([[1, 3, 4], [3, 6, 3]]))
    numbers = tuple_array_to_array(get_numbers(grid))
    logger.info(numbers)
    is_adjacent = find_adjacent_numbers(wildcards, numbers)
    logger.info(is_adjacent.T)
    assert np.all(is_adjacent.T[:4, :] == np.array([[0, 2], [2, 2], [2, 3], [2, 6]]))
    
    # Check what happens with individual adjacent numbers
    single_adjacent_number = np.array([[2], [0],])
    single = get_number_sequences(grid, numbers, single_adjacent_number)  
    
    assert single == [467]
    
    
    number_sequences = get_number_sequences(grid, numbers, np.array([is_adjacent[1], is_adjacent[0]]))
    logger.info(sorted(number_sequences))
    summed = sum(number_sequences)
    logger.info(summed)
    assert summed == 4361
    
def test_alternative():
    import numpy as np
    # Example usage
    schematic = "467..114..\n...*......\n..35..633.\n......#...\n617*......\n.....+.58.\n..592.....\n......755.\n...$.*....\n.664.598.."
    print(sum_part_numbers(schematic))  # Should output the sum of part numbers
    assert sum_part_numbers(schematic) == 4361
    
def test_one_last_go():
    import numpy as np
    # Example usage
    schematic = "467..114..\n...*......\n..35..633.\n......#...\n617*......\n.....+.58.\n..592.....\n......755.\n...$.*....\n.664.598.."
    print(final_go(schematic))  # Should output the sum of part numbers
    # assert one_last_go(schematic) == 4361