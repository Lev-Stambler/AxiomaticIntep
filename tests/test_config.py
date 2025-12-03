"""Tests for configuration module."""

import pytest

from interventionfeatures.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    DatabaseConfig,
    EarlyStoppingConfig,
    generate_param_hash,
    setup_device,
    set_global_seed,
)


class TestConfig:
    """Test cases for Config dataclass."""

    def test_default_values(self):
        """Test that Config initializes with expected default values."""
        config = Config()

        assert config.seed == 62
        assert config.output_dir == "outputs"
        assert config.model.layer_cutoff == 2
        assert config.model.model_name == "EleutherAI/pythia-70m-deduped"
        assert config.model.max_seq_len == 40
        assert config.training.pga_batch_size == 64
        assert config.training.dict_size == 1
        assert config.training.learning_rate == 0.2

    def test_model_config(self):
        """Test ModelConfig dataclass."""
        model_config = ModelConfig()

        assert model_config.model_name == "EleutherAI/pythia-70m-deduped"
        assert model_config.layer_cutoff == 2
        assert model_config.hook_type == "hook_resid_post"
        assert model_config.target_norm == 10.0

    def test_training_config(self):
        """Test TrainingConfig dataclass."""
        training_config = TrainingConfig()

        assert training_config.pga_batch_size == 64
        assert training_config.eval_batch_size == 128
        assert training_config.learning_rate == 0.2
        assert training_config.early_stopping.enabled is False

    def test_early_stopping_config(self):
        """Test EarlyStoppingConfig dataclass."""
        es_config = EarlyStoppingConfig()

        assert es_config.enabled is False
        assert es_config.patience == 5
        assert es_config.min_delta == 1e-6

    def test_get_dict(self):
        """Test that get_dict returns a dictionary representation."""
        config = Config()
        config_dict = config.get_dict()

        assert isinstance(config_dict, dict)
        assert "seed" in config_dict
        assert "model" in config_dict
        assert config_dict["seed"] == 62


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

    def test_setup_device(self):
        """Test device setup function."""
        device = setup_device()
        assert device in ["cuda", "mps", "cpu"]
        assert isinstance(device, str)

    def test_set_global_seed(self):
        """Test that set_global_seed doesn't raise errors."""
        # Should not raise any exceptions
        set_global_seed(42)
        set_global_seed(0)


if __name__ == "__main__":
    pytest.main([__file__])
