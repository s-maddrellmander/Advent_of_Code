from solutions.year_2024.day_09 import *
import pytest

def test_unpack_input():
    data = ['12345']
    
    res, spaces = unpack_input(data)
    assert res == [0, None, None, 1, 1, 1, None, None, None, None, 2, 2, 2, 2, 2]
    assert spaces == {1:2, 6:4}

    
def test_unpack_full():
    data = ['2333133121414131402']
    
    expected = [0,0,None,None,None,1,1,1,None,None,None,2,None,None,None,3,3,3,None,4,4,None,5,5,5,5,None,6,6,6,6,None,7,7,7,None,8,8,8,8,9,9]
    assert unpack_input(data)[0] == expected 

def test_compress():
    data = [0, None, None, 1, 1, 1, None, None, None, None, 2, 2, 2, 2, 2]
    
    res = compress(data)
    assert res == [0, 2, 2, 1, 1, 1, 2, 2, 2, None, None, None, None, None, None]

def test_compute_checksum():
    data = [0, 0, 9, 9]
    
    res = compute_checksum(data)
    assert res == 0 + 0 + 18 + 27

def test_part1():
    data = ['2333133121414131402']
    
    res = part1(data)
    assert res == 1928

def test_part2():
    data = ['2333133121414131402']
    
    res = part2(data)
    assert res == 2858