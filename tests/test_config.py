"""Tests for configuration module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.interventionfeatures.utils.config import (
    MainConfig,
    generate_param_hash,
    setup_device,
)


class TestMainConfig:
    """Test cases for MainConfig class."""

    def test_init_default_values(self):
        """Test that MainConfig initializes with expected default values."""
        config = MainConfig()

        assert config.layer_cutoff == 2
        assert config.model_name == "EleutherAI/pythia-70m-deduped"
        assert config.seed == 62
        assert config.target_norm == 10.0
        assert config.max_seq_len == 40
        assert config.pga_batch_size == 64
        assert config.dict_size == 1
        assert config.learning_rate == 0.2
        assert config.use_ax is False

    def test_get_db_params_dict(self):
        """Test that get_db_params_dict returns expected parameters."""
        config = MainConfig()
        db_params = config.get_db_params_dict()

        expected_keys = {
            "model_name",
            "layer_cutoff",
            "hook_type",
            "max_seq_len",
            "index_size",
            "index_batch_size",
        }
        assert set(db_params.keys()) == expected_keys
        assert db_params["model_name"] == config.model_name
        assert db_params["layer_cutoff"] == config.layer_cutoff

    def test_get_data_params_dict(self):
        """Test that get_data_params_dict returns expected parameters."""
        config = MainConfig()
        data_params = config.get_data_params_dict()

        # Should contain all training and model parameters
        assert "model_name" in data_params
        assert "pga_its" in data_params
        assert "learning_rate" in data_params
        assert "dict_size" in data_params

    def test_validate_valid_config(self):
        """Test that validate() passes for valid configuration."""
        config = MainConfig()
        # Should not raise any exception
        config.validate()

    def test_validate_invalid_config(self):
        """Test that validate() raises errors for invalid configuration."""
        config = MainConfig()

        # Test negative layer_cutoff
        config.layer_cutoff = -1
        with pytest.raises(ValueError, match="layer_cutoff must be non-negative"):
            config.validate()

        # Reset and test invalid max_seq_len
        config = MainConfig()
        config.max_seq_len = 0
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            config.validate()

        # Reset and test invalid learning_rate
        config = MainConfig()
        config.learning_rate = -0.1
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

    def test_save_and_load_config(self):
        """Test saving and loading configuration to/from file."""
        config = MainConfig()
        config.model_name = "test-model"
        config.learning_rate = 0.1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = Path(f.name)

        try:
            # Save config
            config.save_to_file(config_path)
            assert config_path.exists()

            # Load config
            loaded_config = MainConfig.from_file(config_path)
            assert loaded_config.model_name == "test-model"
            assert loaded_config.learning_rate == 0.1

        finally:
            if config_path.exists():
                config_path.unlink()

    def test_load_nonexistent_config(self):
        """Test loading from nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            MainConfig.from_file("nonexistent_config.json")


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_generate_param_hash(self):
        """Test parameter hash generation."""
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}  # Same content, different order
        params3 = {"a": 1, "b": 3}  # Different content

        hash1 = generate_param_hash(params1)
        hash2 = generate_param_hash(params2)
        hash3 = generate_param_hash(params3)

        # Same parameters should produce same hash regardless of order
        assert hash1 == hash2
        # Different parameters should produce different hash
        assert hash1 != hash3

        # Test with prefix
        hash_with_prefix = generate_param_hash(params1, prefix="test_")
        assert hash_with_prefix.startswith("test_")
        assert hash_with_prefix.endswith(hash1)

    def test_generate_param_hash_empty_dict(self):
        """Test that empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="Parameters dictionary cannot be empty"):
            generate_param_hash({})

        with pytest.raises(ValueError, match="Parameters dictionary cannot be empty"):
            generate_param_hash(None)

    def test_setup_device(self):
        """Test device setup function."""
        device = setup_device()
        assert device in ["cuda", "mps", "cpu"]
        assert isinstance(device, str)


if __name__ == "__main__":
    pytest.main([__file__])
