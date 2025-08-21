"""Pytest configuration and shared fixtures."""

# Fix sqlite3 module for ChromaDB compatibility before any other imports
try:
    import sys

    import pysqlite3

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # pysqlite3 not available, continue with system sqlite3
    pass

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.interventionfeatures.utils.config import set_global_seed

from src.interventionfeatures.utils.config import MainConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = MainConfig()
    config.model_name = "test-model"
    config.layer_cutoff = 1
    config.max_seq_len = 10
    config.pga_batch_size = 4
    config.dict_size = 1
    return config


@pytest.fixture
def mock_data_handler():
    """Create a mock data handler for testing."""
    handler = MagicMock()
    handler.d_model = 32
    handler.model = MagicMock()
    handler.tokenizer = MagicMock()
    handler.hook_point_name_for_x = "test_hook"
    return handler


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(4, 32)


@pytest.fixture
def sample_activations():
    """Create sample activation data for testing."""
    return {
        "activations": torch.randn(8, 32),
        "tokens": torch.randint(0, 1000, (8,)),
        "sample_idx": 0,
    }


@pytest.fixture(autouse=True)
def set_torch_deterministic():
    """Set global random seeds and PyTorch to deterministic mode for reproducible tests."""
    set_global_seed(42)


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client."""
    client = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 0
    client.get_or_create_collection.return_value = collection
    client.create_collection.return_value = collection
    return client


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
