import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from data_parser import load_file


# @pytest.mark.skip(reason="Not implemented yet")
def test_load_file_success(mocker):
    mocker.patch("builtins.open", mocker.mock_open(read_data="line1\nline2\nline3\n"))
    result = load_file("testfile")
    assert result == ["line1", "line2", "line3"]


# @pytest.mark.skip(reason="Not implemented yet")
def test_load_file_not_found(mocker):
    mocker.patch("builtins.open", side_effect=FileNotFoundError())
    result = load_file("nonexistentfile")
    assert result == []


# @pytest.mark.skip(reason="Not implemented yet")
def test_load_file_error(mocker):
    mocker.patch("builtins.open", side_effect=Exception("Unexpected error"))
    result = load_file("errorfile")
    assert result == []


def test_with_real_file():
    cwd = Path(__file__).parent
    # Combine the current working directory with the filename
    filename = cwd / "day_01.txt"

    result = load_file(filename)
    assert len(result) == 4
