"""Base class for MIB task implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseMIBTask(ABC):
    """
    Abstract base class for MIB task implementations.

    Each task evaluates Interchange Intervention Accuracy (IIA):
    - Feed model base input, extract activations at intervention point
    - Apply featurizer to swap features with source input
    - Check if output matches expected behavior under intervention

    Subclasses must implement:
        - get_token_position(): Position to intervene
        - get_expected_output(): Expected output given intervention
        - process_example(): Convert dataset example to standard format
    """

    def __init__(
        self,
        model,
        layer: int,
        device: str = "cuda",
    ):
        """
        Initialize MIB task.

        Args:
            model: HookedTransformer model
            layer: Layer index for interventions
            device: Device to run on
        """
        self.model = model
        self.layer = layer
        self.device = device
        self.hook_name = f"blocks.{layer}.hook_resid_post"

    @abstractmethod
    def get_token_position(self, example: dict) -> int:
        """
        Return the token position to intervene on.

        Args:
            example: Processed example dictionary

        Returns:
            Token position index (can be negative for from-end indexing)
        """
        pass

    @abstractmethod
    def get_expected_output(self, example: dict, intervened: bool) -> str:
        """
        Return expected output given intervention status.

        Args:
            example: Processed example dictionary
            intervened: Whether intervention was applied

        Returns:
            Expected output token/string
        """
        pass

    @abstractmethod
    def process_example(self, raw_example: dict) -> dict:
        """
        Convert raw dataset example to standard format.

        Args:
            raw_example: Example from HuggingFace dataset

        Returns:
            Dictionary with keys:
                - base_input: str - Base input text
                - source_input: str - Source input text
                - base_answer: str - Expected answer without intervention
                - source_answer: str - Expected answer after intervention
                - intervention_position: int - Token position (optional)
                - metadata: dict - Additional task-specific data
        """
        pass

    def evaluate_single(
        self,
        example: dict,
        featurizer,
    ) -> float:
        """
        Evaluate IIA for a single example.

        Args:
            example: Processed example dictionary
            featurizer: CSSFeaturizer instance

        Returns:
            1.0 if interchange intervention produces correct output, 0.0 otherwise
        """
        # Get base and source inputs
        base_input = example["base_input"]
        source_input = example["source_input"]

        # Tokenize
        base_tokens = self.model.tokenizer(
            base_input, return_tensors="pt", padding=True
        ).input_ids.to(self.device)

        source_tokens = self.model.tokenizer(
            source_input, return_tensors="pt", padding=True
        ).input_ids.to(self.device)

        # Get position to intervene
        pos = self.get_token_position(example)

        # Handle negative indexing
        if pos < 0:
            pos = base_tokens.shape[1] + pos

        # Get activations at hook point
        with torch.no_grad():
            _, base_cache = self.model.run_with_cache(base_tokens, names_filter=[self.hook_name])
            _, source_cache = self.model.run_with_cache(
                source_tokens, names_filter=[self.hook_name]
            )

        base_act = base_cache[self.hook_name][:, pos, :]  # (1, d_model)
        source_act = source_cache[self.hook_name][:, pos, :]

        # Apply featurizer intervention
        intervened_act = featurizer(base_act, source_act)

        # Create hook to inject intervened activation
        def intervention_hook(activation, hook):
            activation[:, pos, :] = intervened_act
            return activation

        # Run with intervention
        with self.model.hooks(fwd_hooks=[(self.hook_name, intervention_hook)]):
            output_logits = self.model(base_tokens)

        # Get prediction
        predicted_id = output_logits[0, -1].argmax().item()
        predicted_token = self.model.tokenizer.decode([predicted_id]).strip()

        # Check against expected output
        expected = self.get_expected_output(example, intervened=True)

        return self._check_match(predicted_token, expected)

    def _check_match(self, predicted: str, expected: str) -> float:
        """
        Check if prediction matches expected output.

        Args:
            predicted: Model's predicted token
            expected: Expected output

        Returns:
            1.0 if match, 0.0 otherwise
        """
        pred_clean = predicted.strip().lower()
        exp_clean = expected.strip().lower()

        # Exact match
        if pred_clean == exp_clean:
            return 1.0

        # Partial match (expected contained in predicted or vice versa)
        if exp_clean in pred_clean or pred_clean in exp_clean:
            return 1.0

        return 0.0

    def evaluate_batch(
        self,
        examples: list[dict],
        featurizer,
    ) -> list[float]:
        """
        Evaluate IIA for a batch of examples.

        Args:
            examples: List of processed examples
            featurizer: CSSFeaturizer instance

        Returns:
            List of IIA scores (1.0 or 0.0 for each example)
        """
        return [self.evaluate_single(ex, featurizer) for ex in examples]
