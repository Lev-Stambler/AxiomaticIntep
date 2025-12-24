"""Tests for benchmark runners."""

import pytest
import torch
import unittest.mock as mock
from interventionfeatures.benchmarks import MIBBenchmarkRunner, RAVELBenchmarkRunner, BenchmarkResult


class TestBenchmarkRunners:
    """Test cases for MIB and RAVEL benchmark runners."""

    def setup_method(self):
        self.css_directions = [
            {"s": torch.randn(1, 128), "polarity": "positive", "feature_id": "0_positive"},
            {"s": torch.randn(1, 128), "polarity": "positive", "feature_id": "1_positive"}
        ]
        self.model_name = "pythia-70m"
        self.layer = 2

    def test_mib_runner_init(self):
        """Test MIB runner initialization."""
        runner = MIBBenchmarkRunner(
            css_directions=self.css_directions,
            model_name=self.model_name,
            layer=self.layer,
            device="cpu"
        )
        assert runner.model_name == self.model_name
        assert runner.layer == self.layer
        assert runner.device == "cpu"
        assert "ioi" in runner.get_available_tasks()

    def test_ravel_runner_init(self):
        """Test RAVEL runner initialization."""
        runner = RAVELBenchmarkRunner(
            css_directions=self.css_directions,
            model_name=self.model_name,
            layer=self.layer,
            device="cpu"
        )
        assert runner.model_name == self.model_name
        assert runner.layer == self.layer
        assert "cities_Country" in runner.get_available_tasks()

    @mock.patch("interventionfeatures.benchmarks.mib.runner.LMPipeline", create=True)
    @mock.patch("interventionfeatures.benchmarks.mib.runner.PatchResidualStream", create=True)
    @mock.patch("interventionfeatures.benchmarks.mib.runner.Featurizer", create=True)
    def test_mib_run_evaluation(self, mock_featurizer, mock_patch_stream, mock_pipeline):
        """Test MIB run_evaluation with mocking."""
        runner = MIBBenchmarkRunner(
            css_directions=self.css_directions,
            model_name=self.model_name,
            layer=self.layer,
            device="cpu"
        )
        
        # Setup mocks
        mock_instance = mock_patch_stream.return_value
        mock_instance.perform_interventions.return_value = {
            "dataset": {
                "test_ds": {
                    "model_unit": {
                        "unit": {
                            "indirect_object": {"average_score": 0.85}
                        }
                    }
                }
            }
        }
        
        # Mock task modules
        with mock.patch.object(runner, "_get_task_modules") as mock_modules:
            mock_cf_datasets = mock.Mock()
            mock_cf_datasets.return_value = {"test_data": mock.Mock()}
            mock_token_pos = mock.Mock()
            mock_token_pos.return_value = [mock.Mock(id="pos1")]
            mock_modules.return_value = (
                mock_cf_datasets, # get_cf_datasets
                mock.Mock(), # get_causal_model
                mock_token_pos, # get_token_pos
                "indirect_object" # variable
            )
            
            result = runner.run_evaluation("ioi", num_samples=10)
            
            assert isinstance(result, BenchmarkResult)
            assert result.task_name == "ioi"
            assert result.metrics["iia"] == 0.85

    @mock.patch("transformer_lens.HookedTransformer")
    def test_ravel_run_evaluation_synthetic(self, mock_transformer):
        """Test RAVEL run_evaluation with synthetic data fallback."""
        runner = RAVELBenchmarkRunner(
            css_directions=self.css_directions,
            model_name=self.model_name,
            layer=self.layer,
            device="cpu",
            ravel_repo_path="/non/existent/path"
        )
        
        # Mock model behaviors
        mock_model = mock_transformer.from_pretrained.return_value
        mock_model.tokenizer.return_value = mock.Mock(input_ids=torch.zeros((1, 5)))
        mock_model.tokenizer.decode.return_value = " Paris"
        mock_model.return_value = torch.randn((1, 5, 50000)) # Logits
        
        # We need to mock _get_activations and _run_with_intervention since they use HookedTransformer
        with mock.patch.object(runner, "_get_activations") as mock_act, \
             mock.patch.object(runner, "_run_with_intervention") as mock_run:
            
            mock_act.return_value = torch.randn((1, 128))
            mock_run.return_value = torch.randn((1, 1, 50000))
            
            result = runner.run_evaluation("cities_country", num_samples=2)
            
            assert isinstance(result, BenchmarkResult)
            assert "cause" in result.metrics
            assert "isolation" in result.metrics
