"""Pytest configuration and shared fixtures."""

import sys

# Fix sqlite3 module for ChromaDB compatibility
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from interventionfeatures.config import Config, set_global_seed


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return Config()


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


@pytest.fixture(autouse=True)
def set_torch_deterministic():
    """Set global random seeds for reproducible tests."""
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
