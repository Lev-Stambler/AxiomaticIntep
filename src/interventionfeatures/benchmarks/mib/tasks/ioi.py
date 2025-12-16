"""Indirect Object Identification (IOI) task for MIB benchmark."""

from __future__ import annotations

from .base import BaseMIBTask


class IOITask(BaseMIBTask):
    """
    Indirect Object Identification task.

    Tests whether the model can identify the indirect object in sentences like:
    "John gave Mary the book. Mary gave the book to"
    Expected: "John"

    The causal variable is the identity of the indirect object (IO),
    and we test if swapping the IO representation causes the model
    to predict the source's IO instead.
    """

    def get_token_position(self, example: dict) -> int:
        """IOI typically intervenes at the S2 (second subject) position."""
        # Use provided position or default to second-to-last
        return example.get("intervention_position", -2)

    def get_expected_output(self, example: dict, intervened: bool) -> str:
        """Return IO name from source if intervened."""
        if intervened:
            return example["source_answer"]
        return example["base_answer"]

    def process_example(self, raw_example: dict) -> dict:
        """
        Convert IOI dataset example to standard format.

        IOI examples typically have:
        - text: The full sentence
        - IO: Indirect object name
        - S: Subject name
        """
        # Handle different possible formats
        if "base_input" in raw_example:
            # Already in standard format
            return raw_example

        # Convert from MIB format
        return {
            "base_input": raw_example.get("base_text", raw_example.get("text", "")),
            "source_input": raw_example.get(
                "source_text", raw_example.get("counterfactual_text", "")
            ),
            "base_answer": raw_example.get("base_io", raw_example.get("IO", "")),
            "source_answer": raw_example.get("source_io", raw_example.get("counterfactual_IO", "")),
            "intervention_position": raw_example.get("intervention_position", -2),
            "metadata": {
                "subject": raw_example.get("S", ""),
                "task_type": "ioi",
            },
        }
