import pytest

# Now you can import your module
from run import main, run_solution


@pytest.mark.skip(reason="Not implemented yet")
def test_main_with_no_args(capsys, monkeypatch):
    # Mock sys.argv to simulate no additional arguments
    monkeypatch.setattr("sys.argv", ["run.py"])

    # Call main
    main()

    # Capture the output
    captured = capsys.readouterr()

    # Check that the usage message was printed
    assert "Usage: python run.py <day_number> [part_number]" in captured.out


@pytest.mark.skip(reason="Not implemented yet")
def test_main_with_args(capsys, monkeypatch):
    # Mock sys.argv to simulate command line arguments
    monkeypatch.setattr("sys.argv", ["run.py", "0", "1"])

    # Call main
    main()

    # Capture the output
    captured = capsys.readouterr()

    # Check that the usage message was printed
    assert "Part 1 solution not implemented." in captured.out
