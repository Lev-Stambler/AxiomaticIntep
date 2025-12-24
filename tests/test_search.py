"""Tests for search module with Open-Puffer backend."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from interventionfeatures.core.search import (
    ActivationSimSearcher,
    _find_optimal_ordered,
)


class TestOpenPufferClient:
    """Test cases for OpenPufferClient."""

    def test_health_check_success(self):
        """Test health check when server is available."""
        from interventionfeatures.core.openpuffer_client import OpenPufferClient

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            client = OpenPufferClient(host="localhost", port=8080)
            assert client.health_check() is True

    def test_health_check_failure(self):
        """Test health check when server is unavailable."""
        from interventionfeatures.core.openpuffer_client import OpenPufferClient

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("Connection refused")
            mock_session_class.return_value = mock_session

            client = OpenPufferClient(host="localhost", port=8080)
            assert client.health_check() is False

    def test_create_collection(self):
        """Test collection creation."""
        from interventionfeatures.core.openpuffer_client import OpenPufferClient

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "test", "dimension": 64}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            client = OpenPufferClient()
            result = client.create_collection("test", 64, "cosine")

            assert result["name"] == "test"
            mock_session.post.assert_called_once()

    def test_insert_points(self):
        """Test inserting points."""
        import numpy as np

        from interventionfeatures.core.openpuffer_client import OpenPufferClient

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"inserted": 2}
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            client = OpenPufferClient()
            vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            result = client.insert_points(
                collection_name="test",
                ids=["id1", "id2"],
                vectors=vectors,
                payloads=[{"a": 1}, {"a": 2}],
            )

            assert result["inserted"] == 2

    def test_search(self):
        """Test searching."""
        import numpy as np

        from interventionfeatures.core.openpuffer_client import OpenPufferClient

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"id": "id1", "score": 0.9, "payload": {"sample_idx": 0}},
                    {"id": "id2", "score": 0.8, "payload": {"sample_idx": 1}},
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            client = OpenPufferClient()
            query = np.array([1.0, 2.0], dtype=np.float32)
            result = client.search("test", query, top_k=2)

            assert len(result["results"]) == 2
            assert result["results"][0]["score"] == 0.9


class TestActivationSimSearcherMocked:
    """Test cases for ActivationSimSearcher with mocked Open-Puffer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_handler = MagicMock()
        self.mock_data_handler.d_model = 64
        self.mock_data_handler.max_seq_len = 40
        self.device = "cpu"
        self.d_model = 64

    @patch("interventionfeatures.core.search.get_openpuffer_client")
    def test_init_with_valid_scoring_type(self, mock_get_client):
        """Test initialization with valid scoring types."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = {}
        mock_get_client.return_value = (mock_client, None)

        # Test cosine scoring
        searcher_cosine = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="cosine",
        )
        assert searcher_cosine.scoring_type == "cosine"
        assert searcher_cosine.metric == "cosine"

        # Test dot scoring
        searcher_dot = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )
        assert searcher_dot.scoring_type == "dot"
        assert searcher_dot.metric == "dot"

    @patch("interventionfeatures.core.search.get_openpuffer_client")
    def test_init_with_invalid_scoring_type(self, mock_get_client):
        """Test that invalid scoring type raises ValueError."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, None)

        with pytest.raises(ValueError, match="scoring_type must be 'cosine' or 'dot'"):
            ActivationSimSearcher(
                data_handler=self.mock_data_handler,
                device=self.device,
                d_model=self.d_model,
                scoring_type="invalid",
            )

    @patch("interventionfeatures.core.search.get_openpuffer_client")
    def test_normalize_if_cosine(self, mock_get_client):
        """Test vector normalization for cosine similarity."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = {}
        mock_get_client.return_value = (mock_client, None)

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

    @patch("interventionfeatures.core.search.get_openpuffer_client")
    def test_normalize_if_dot(self, mock_get_client):
        """Test that dot product scoring doesn't normalize."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = {}
        mock_get_client.return_value = (mock_client, None)

        searcher_dot = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )

        vectors = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
        unchanged = searcher_dot._normalize_if_cosine(vectors)
        assert torch.equal(vectors, unchanged)

    @patch("interventionfeatures.core.search.get_openpuffer_client")
    def test_is_indexed_initially_false(self, mock_get_client):
        """Test that is_indexed is False initially."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = {}
        mock_get_client.return_value = (mock_client, None)

        searcher = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )

        assert searcher.is_indexed is False
        assert searcher.indexed_sample_tokens == []

    @patch("interventionfeatures.core.search.get_openpuffer_client")
    def test_get_index_stats_not_indexed(self, mock_get_client):
        """Test get_index_stats when not indexed."""
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = {}
        mock_get_client.return_value = (mock_client, None)

        searcher = ActivationSimSearcher(
            data_handler=self.mock_data_handler,
            device=self.device,
            d_model=self.d_model,
            scoring_type="dot",
        )

        stats = searcher.get_index_stats()
        assert stats["indexed"] is False


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

    def test_find_optimal_ordered_sum_score(self):
        """Test that sum scoring returns expected results."""
        # Tensor where optimal path is clear
        tensor = torch.tensor([[10.0, 1.0, 1.0], [1.0, 1.0, 10.0]])

        indices, score = _find_optimal_ordered(tensor, "sum")

        # Optimal should be (0, 2) with score 20
        assert indices[0] == 0
        assert indices[1] == 2
        assert score == 20.0

    def test_find_optimal_ordered_mean_score(self):
        """Test that mean scoring returns expected results."""
        tensor = torch.tensor([[10.0, 1.0, 1.0], [1.0, 1.0, 10.0]])

        indices, score = _find_optimal_ordered(tensor, "mean")

        # Optimal should be (0, 2) with mean score 10
        assert indices[0] == 0
        assert indices[1] == 2
        assert score == 10.0


class TestServerManager:
    """Test cases for OpenPufferServerManager."""

    def test_init(self):
        """Test server manager initialization."""
        from interventionfeatures.core.openpuffer_client import OpenPufferServerManager

        manager = OpenPufferServerManager(
            binary_path="/path/to/puffer-server",
            data_dir="/path/to/data",
            port=8080,
        )

        assert manager.binary_path == "/path/to/puffer-server"
        assert manager.data_dir == "/path/to/data"
        assert manager.port == 8080
        assert manager.process is None

    def test_is_running_when_not_started(self):
        """Test is_running when server not started."""
        from interventionfeatures.core.openpuffer_client import OpenPufferServerManager

        manager = OpenPufferServerManager(
            binary_path="/path/to/puffer-server",
            data_dir="/path/to/data",
        )

        assert manager.is_running() is False


if __name__ == "__main__":
    pytest.main([__file__])
