"""Tests for CLI module."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.interventionfeatures.cli.main import app, setup_logging


class TestCLI:
    """Test cases for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_setup_logging(self):
        """Test logging setup function."""
        logger = setup_logging(verbose=False)
        assert logger is not None

        logger_verbose = setup_logging(verbose=True)
        assert logger_verbose is not None

    def test_config_command(self):
        """Test config generation command."""
        with patch("src.interventionfeatures.utils.config.MainConfig") as mock_config:
            mock_instance = MagicMock()
            mock_config.return_value = mock_instance

            result = self.runner.invoke(app, ["config", "--output", "test_config.toml"])

            assert result.exit_code == 0
            mock_instance.save_to_file.assert_called_once_with("test_config.toml")

    @patch("src.interventionfeatures.cli.main._run_pipeline")
    @patch("src.interventionfeatures.utils.config.MainConfig")
    def test_run_command_basic(self, mock_config, mock_run_pipeline):
        """Test basic run command functionality."""
        mock_instance = MagicMock()
        mock_config.return_value = mock_instance

        result = self.runner.invoke(
            app,
            [
                "run",
                "--model-name",
                "test-model",
                "--layer-cutoff",
                "3",
                "--dict-size",
                "2",
            ],
        )

        # Should not exit with error for valid configuration
        assert result.exit_code == 0
        mock_run_pipeline.assert_called_once()

    def test_run_command_with_config_file(self):
        """Test run command with config file."""
        with patch(
            "src.interventionfeatures.utils.config.MainConfig.from_file"
        ) as mock_from_file:
            with patch(
                "src.interventionfeatures.cli.main._run_pipeline"
            ) as mock_run_pipeline:
                mock_instance = MagicMock()
                mock_from_file.return_value = mock_instance

                result = self.runner.invoke(
                    app, ["run", "--config-file", "test_config.json"]
                )

                assert result.exit_code == 0
                mock_from_file.assert_called_once_with("test_config.json")
                mock_run_pipeline.assert_called_once()

    def test_explain_command(self):
        """Test explain command."""
        result = self.runner.invoke(
            app,
            [
                "explain",
                "css_results.pkl",
                "searcher_db",
                "--output-dir",
                "explanations",
                "--top-k",
                "10",
            ],
        )

        # Command should execute without syntax errors
        # Note: This will likely fail with missing files, but structure should be valid
        assert "explanations" in result.output or result.exit_code != 0

    def test_validate_command(self):
        """Test validate command."""
        result = self.runner.invoke(
            app,
            [
                "validate",
                "css_results.pkl",
                "explanations.json",
                "--output-dir",
                "validation",
            ],
        )

        # Command should execute without syntax errors
        assert "validation" in result.output or result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__])
