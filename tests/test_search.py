"""Tests for search module."""

# Fix sqlite3 module for ChromaDB compatibility
try:
    import sys

    import pysqlite3

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # pysqlite3 not available, continue with system sqlite3
    pass

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from interventionfeatures.core.search import (
    ActivationSimSearcher,
    _find_optimal_ordered,
)


class TestActivationSimSearcher:
    """Test cases for ActivationSimSearcher class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_handler = MagicMock()
        self.mock_data_handler.d_model = 64
        self.device = "cpu"
        self.d_model = 64

    def test_init_with_valid_scoring_type(self):
        """Test initialization with valid scoring types."""
        # Test cosine scoring
        searcher_cosine = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="cosine",
        )
        assert searcher_cosine.scoring_type == "cosine"
        assert searcher_cosine.space == "cosine"

        # Test dot scoring
        searcher_dot = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )
        assert searcher_dot.scoring_type == "dot"
        assert searcher_dot.space == "ip"

    def test_init_with_invalid_scoring_type(self):
        """Test that invalid scoring type raises ValueError."""
        with pytest.raises(ValueError, match="scoring_type must be 'cosine' or 'dot'"):
            ActivationSimSearcher(
                data_handler=self.mock_data_handler,
                device=self.device,
                d_model=self.d_model,
                scoring_type="invalid",
            )

    @patch("chromadb.Client")
    def test_init_without_chroma_db_path(self, mock_client_class):
        """Test initialization without persistent ChromaDB path."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        searcher = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )

        assert searcher.db_path is None
        assert searcher.indexed_sample_tokens == []
        mock_client_class.assert_called_once()

    def test_normalize_if_cosine(self):
        """Test vector normalization for cosine similarity."""
        searcher = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="cosine",
        )

        # Test vector normalization
        vectors = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
        normalized = searcher._normalize_if_cosine(vectors)

        # Check if normalized (L2 norm should be 1)
        norms = torch.norm(normalized, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

        # Test with dot product (should not normalize)
        searcher_dot = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )

        unchanged = searcher_dot._normalize_if_cosine(vectors)
        assert torch.equal(vectors, unchanged)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_find_optimal_ordered_basic(self):
        """Test basic functionality of _find_optimal_ordered."""
        # Simple test case
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        indices, score = _find_optimal_ordered(tensor, "sum")

        # Should return valid indices and a score
        assert indices.shape == (2,)
        assert isinstance(score, float)
        assert torch.all(indices >= 0)
        assert torch.all(indices < tensor.shape[1])

        # For ordered sequence, indices should be ordered
        assert indices[0] < indices[1]

    def test_find_optimal_ordered_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        # K > SEQ_LEN should raise error
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # K=3, SEQ_LEN=2

        with pytest.raises(
            ValueError, match="SEQ_LEN .* must be greater than or equal to K"
        ):
            _find_optimal_ordered(tensor, "sum")

    def test_find_optimal_ordered_different_scoring_functions(self):
        """Test different scoring functions."""
        tensor = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])

        # Test different scoring functions
        for score_func in ["sum", "mean", "max", "min"]:
            indices, score = _find_optimal_ordered(tensor, score_func)
            assert indices.shape == (2,)
            assert isinstance(score, float)


if __name__ == "__main__":
    pytest.main([__file__])
