"""Tests for CLI module."""

import pytest

from interventionfeatures.cli.main import main, print_config


class TestCLI:
    """Test cases for CLI functionality."""

    def test_print_config(self, capsys):
        """Test that print_config outputs YAML."""
        print_config()
        captured = capsys.readouterr()
        assert "seed" in captured.out
        assert "model" in captured.out
        assert "training" in captured.out

    def test_main_help(self, capsys):
        """Test that main shows help with no arguments."""
        import sys
        sys.argv = ["interventionfeatures"]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


if __name__ == "__main__":
    pytest.main([__file__])
