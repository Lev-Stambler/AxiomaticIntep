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
    @mock.patch("transformer_lens.HookedTransformer")
    def test_mib_run_evaluation(self, mock_transformer, mock_featurizer, mock_patch_stream, mock_pipeline):
        """Test MIB run_evaluation with mocking."""
        runner = MIBBenchmarkRunner(
            css_directions=self.css_directions,
            model_name=self.model_name,
            layer=self.layer,
            device="cpu"
        )

        # Mock model
        mock_model = mock_transformer.from_pretrained.return_value
        mock_model.run_with_cache.return_value = (None, {"blocks.2.hook_resid_post": torch.randn(1, 5, 128)})
        mock_model.generate.return_value = torch.zeros((1, 6), dtype=torch.long)  # 5 input + 1 output

        # Mock pipeline
        mock_pipe_instance = mock_pipeline.return_value
        mock_pipe_instance.model = mock_model
        mock_pipe_instance.tokenizer.decode.return_value = " result"
        mock_pipe_instance.tokenizer.eos_token_id = 0
        mock_pipe_instance.load.return_value = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}
        mock_pipe_instance.max_new_tokens = 1

        # Create mock functions that return their values correctly
        def mock_get_cf_datasets(hf=True, size=None):
            return {"test_ds": [{"example": 1}]}  # Dict with "test" key

        mock_causal_model = mock.Mock()
        mock_causal_model.label_counterfactual_data.return_value = [
            {
                "input": "Base input",
                "counterfactual_inputs": ["Source input"],
                "label": "result"
            }
        ]

        def mock_get_causal_model():
            return mock_causal_model

        mock_token_indexer = mock.Mock()
        mock_token_indexer.index.return_value = 0

        def mock_get_token_pos(pipeline, causal_model):
            return [mock_token_indexer]

        # Mock task modules
        with mock.patch.object(runner, "_get_task_modules") as mock_modules:
            mock_modules.return_value = (
                mock_get_cf_datasets,
                mock_get_causal_model,
                mock_get_token_pos,
                "raw_output"
            )

            # Disable filtering in test since we're mocking the model
            result = runner.run_evaluation("ioi", num_samples=10, filter_examples=False)

            assert isinstance(result, BenchmarkResult)
            assert result.task_name == "ioi"
            # Score is computed from checker; may be 0 or 1 depending on mock output
            assert "iia" in result.metrics

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
            
            result = runner.run_evaluation("cities_Country", num_samples=2)
            
            assert isinstance(result, BenchmarkResult)
            assert "cause" in result.metrics
            assert "isolation" in result.metrics
