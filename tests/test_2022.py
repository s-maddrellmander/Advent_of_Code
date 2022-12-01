import pytest
from utils import load_file
import jax.numpy as jnp

from year_2022 import (day_1)


def test_grouping():
    raw_list = ["1000","2000","3000","","4000","",
                "5000","6000","","7000","8000","9000",
                "","10000"] 
    formatted = day_1.grouping(raw_list)
    assert len(formatted) == 5
    assert isinstance(formatted[0], jnp.ndarray)

@pytest.mark.parametrize("input,expected", [(["1000","2000","3000","","4000","",
                                             "5000","6000","","7000","8000","9000",
                                             "","10000"], 24000)])
def test_day_1_1(input, expected):
    input = day_1.grouping(input)
    result = day_1.part_1(input)
    assert result == expected


@pytest.mark.parametrize("input,expected", [(["1000","2000","3000","","4000","",
                                             "5000","6000","","7000","8000","9000",
                                             "","10000"], 45000)])
def test_day_1_2(input, expected):
    input = day_1.grouping(input)
    result = day_1.part_2(input)
    assert result == expected