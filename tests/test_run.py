import importlib
import logging

from data_parser import load_file
from run import main, run_solution


def test_main_with_specific_part(caplog, monkeypatch):
    # Set the log level you want to capture
    caplog.set_level(logging.INFO)

    # Use monkeypatch to simulate command line arguments
    monkeypatch.setattr("sys.argv", ["run.py", "0", "1"])

    # Run the function which will produce log output
    main()

    # Use caplog to assert log messages
    assert "⭐ Part 1: Timer started." in caplog.text
    assert "⭐ Part 2: Timer started." not in caplog.text


def test_main_with_both_parts(caplog, monkeypatch):
    # Set the log level you want to capture
    caplog.set_level(logging.INFO)

    # Use monkeypatch to simulate command line arguments
    monkeypatch.setattr("sys.argv", ["run.py", "0"])

    # Run the function which will produce log output
    main()

    # Use caplog to assert log messages
    assert "⭐ Part 1: Timer started." in caplog.text
    assert "⭐ Part 2: Timer started." in caplog.text


def test_run_solution_with_part_1(caplog):
    # Set the log level you want to capture
    caplog.set_level(logging.INFO)
    module = importlib.import_module("solutions.year_2023.day_01")
    data = load_file("inputs/year_2023/day_01.txt")
    run_solution(module, 1, data)
    captured = caplog.text
    assert "⭐ Part 1: Timer started." in captured


def test_run_solution_with_part_2(caplog):
    module = importlib.import_module("solutions.year_2023.day_01")
    data = load_file("inputs/year_2023/day_01.txt")
    run_solution(module, 2, data)
    captured = caplog.text
    assert "⭐ Part 2: Timer started." in captured


def test_main_with_wrong_part(caplog, monkeypatch):
    # Set the log level you want to capture
    caplog.set_level(logging.INFO)

    # Use monkeypatch to simulate command line arguments
    monkeypatch.setattr("sys.argv", ["run.py", "0", "3"])

    # Run the function which will produce log output
    main()

    # Use caplog to assert log messages
    assert (
        "day_00 part 3 not found."
        in caplog.text
    )


def test_main_with_wrong_num_args(caplog, monkeypatch):
    # Set the log level you want to capture
    caplog.set_level(logging.INFO)

    # Use monkeypatch to simulate command line arguments
    monkeypatch.setattr("sys.argv", ["run.py", "0", "1", "0"])

    # Run the function which will produce log output
    main()

    # Use caplog to assert log messages
    assert "Usage: python run.py <day_number> [part_number]" in caplog.text


def test_main_with_wrong_day_module_not_found(caplog, monkeypatch):
    # Set the log level you want to capture
    caplog.set_level(logging.INFO)

    # Use monkeypatch to simulate command line arguments
    monkeypatch.setattr("sys.argv", ["run.py", "99", "1"])

    # Run the function which will produce log output
    main()

    # Use caplog to assert log messages
    assert "Module for day_99 not found." in caplog.text
