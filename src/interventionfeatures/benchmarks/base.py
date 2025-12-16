"""Base classes for benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from .featurizer import CSSFeaturizer


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results.

    Attributes:
        task_name: Name of the task/benchmark
        metrics: Dictionary of metric names to values (e.g., {"cause": 0.85, "isolation": 0.72})
        per_sample_results: List of per-sample results for detailed analysis
        metadata: Additional information about the evaluation
    """

    task_name: str
    metrics: dict[str, float]
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"BenchmarkResult(task={self.task_name}, {metrics_str})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "metrics": self.metrics,
            "per_sample_results": self.per_sample_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create from dictionary."""
        return cls(
            task_name=data["task_name"],
            metrics=data["metrics"],
            per_sample_results=data.get("per_sample_results", []),
            metadata=data.get("metadata", {}),
        )


class BaseBenchmarkRunner(ABC):
    """Abstract base class for benchmark runners.

    Provides common functionality for running RAVEL and MIB benchmarks
    on CSS directions.

    Subclasses must implement:
        - load_data(): Load benchmark dataset
        - run_evaluation(): Execute the benchmark evaluation
    """

    def __init__(
        self,
        css_directions: list[dict],
        model_name: str,
        layer: int,
        device: str = "cuda",
    ):
        """
        Initialize benchmark runner.

        Args:
            css_directions: Output from CSSDirectionFinder.find_optimal_s_directions()
                Each dict should have 's' key with direction tensor
            model_name: HuggingFace model name (e.g., "pythia-70m-deduped")
            layer: Layer index for interventions
            device: Device to run on ("cuda", "cpu", etc.)
        """
        self.css_directions = css_directions
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self._featurizer: CSSFeaturizer | None = None
        self._model = None

    @property
    def d_model(self) -> int:
        """Get model hidden dimension from first CSS direction."""
        if not self.css_directions:
            raise ValueError("No CSS directions provided")
        s = self.css_directions[0]["s"]
        # Handle both (K, d_model) and (d_model,) shapes
        return s.shape[-1]

    def create_featurizer(
        self,
        direction_indices: list[int] | None = None,
        orthonormalize: bool = True,
    ) -> CSSFeaturizer:
        """
        Create featurizer from CSS directions.

        Args:
            direction_indices: Which directions to use (by index). None means all.
            orthonormalize: Whether to orthonormalize the basis.

        Returns:
            CSSFeaturizer instance
        """
        return CSSFeaturizer.from_css_results(
            css_results=self.css_directions,
            d_model=self.d_model,
            direction_indices=direction_indices,
            orthonormalize=orthonormalize,
        )

    def get_featurizer(
        self,
        direction_indices: list[int] | None = None,
        orthonormalize: bool = True,
    ) -> CSSFeaturizer:
        """
        Get or create featurizer (cached).

        Args:
            direction_indices: Which directions to use
            orthonormalize: Whether to orthonormalize

        Returns:
            CSSFeaturizer instance
        """
        # Create new featurizer if params changed or not created yet
        if self._featurizer is None:
            self._featurizer = self.create_featurizer(direction_indices, orthonormalize)
        return self._featurizer

    @abstractmethod
    def load_data(self, task_name: str) -> Any:
        """
        Load benchmark dataset for a specific task.

        Args:
            task_name: Name of the task to load data for

        Returns:
            Dataset in task-specific format
        """
        pass

    @abstractmethod
    def run_evaluation(
        self,
        task_name: str,
        direction_indices: list[int] | None = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run benchmark evaluation for a specific task.

        Args:
            task_name: Name of the task to evaluate
            direction_indices: Which CSS directions to use
            **kwargs: Additional task-specific arguments

        Returns:
            BenchmarkResult with metrics and details
        """
        pass

    def run_all_evaluations(
        self,
        direction_indices: list[int] | None = None,
        **kwargs,
    ) -> dict[str, BenchmarkResult]:
        """
        Run all available tasks for this benchmark.

        Args:
            direction_indices: Which CSS directions to use
            **kwargs: Additional arguments passed to run_evaluation

        Returns:
            Dictionary mapping task names to results
        """
        results = {}
        for task_name in self.get_available_tasks():
            results[task_name] = self.run_evaluation(task_name, direction_indices, **kwargs)
        return results

    @abstractmethod
    def get_available_tasks(self) -> list[str]:
        """Return list of available task names for this benchmark."""
        pass

    def get_hook_name(self) -> str:
        """
        Get the hook name for the intervention layer.

        Returns hook name compatible with TransformerLens naming convention.
        """
        return f"blocks.{self.layer}.hook_resid_post"

    def _get_activations(
        self,
        input_ids: torch.Tensor,
        position: int | None = None,
    ) -> torch.Tensor:
        """
        Get activations at the intervention layer.

        Args:
            input_ids: Tokenized input, shape (batch, seq_len)
            position: Token position to extract. None means all positions.

        Returns:
            Activations at hook point. Shape depends on position:
                - If position is None: (batch, seq_len, d_model)
                - If position is int: (batch, d_model)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        hook_name = self.get_hook_name()

        with torch.no_grad():
            _, cache = self._model.run_with_cache(input_ids, names_filter=[hook_name])

        activations = cache[hook_name]

        if position is not None:
            activations = activations[:, position, :]

        return activations

    def _run_with_intervention(
        self,
        input_ids: torch.Tensor,
        intervened_activations: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """
        Run model with intervened activations at specified position.

        Args:
            input_ids: Input tokens, shape (batch, seq_len)
            intervened_activations: Activations to inject, shape (batch, d_model)
            position: Token position to intervene at

        Returns:
            Model output logits, shape (batch, seq_len, vocab_size)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        hook_name = self.get_hook_name()

        def intervention_hook(activation, hook):
            activation[:, position, :] = intervened_activations
            return activation

        with self._model.hooks(fwd_hooks=[(hook_name, intervention_hook)]):
            output = self._model(input_ids)

        return output

    def _load_model(self):
        """Load the transformer model for evaluation."""
        try:
            from transformer_lens import HookedTransformer
        except ImportError as e:
            raise ImportError(
                "transformer_lens is required for benchmarks. "
                "Install with: pip install transformer-lens"
            ) from e

        self._model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
        self._model.eval()
