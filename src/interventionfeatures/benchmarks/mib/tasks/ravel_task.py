"""RAVEL task for MIB benchmark.

This is the MIB version of RAVEL, evaluating IIA metric for consistency
with other MIB tasks. For full RAVEL metrics (CAUSE, Isolation, Disentangle),
use the standalone RAVELBenchmarkRunner.
"""

from __future__ import annotations

from .base import BaseMIBTask


class RAVELTask(BaseMIBTask):
    """
    RAVEL task within MIB framework.

    Tests attribute disentanglement on city entities, evaluating whether
    swapping attribute representations causes the model to output the
    source's attribute value.

    This uses IIA (Interchange Intervention Accuracy) metric for consistency
    with other MIB tasks, rather than RAVEL's native CAUSE/Isolation scores.

    Example:
    Base: "Paris is located in" -> "France"
    Source: "Berlin is located in" -> "Germany"
    After intervention: "Paris is located in" -> "Germany" (IIA = 1.0)
    """

    def __init__(
        self,
        model,
        layer: int,
        device: str = "cuda",
        attribute: str = "country",
    ):
        """
        Initialize RAVEL task.

        Args:
            model: HookedTransformer model
            layer: Layer index for interventions
            device: Device to run on
            attribute: Target attribute (country, language, etc.)
        """
        super().__init__(model, layer, device)
        self.attribute = attribute

    def get_token_position(self, example: dict) -> int:
        """Intervene at entity mention position."""
        return example.get("intervention_position", -2)

    def get_expected_output(self, example: dict, intervened: bool) -> str:
        """Return attribute value based on intervention."""
        if intervened:
            return example["source_answer"]
        return example["base_answer"]

    def process_example(self, raw_example: dict) -> dict:
        """
        Convert RAVEL dataset example to standard format.

        RAVEL examples have:
        - base_prompt / source_prompt: Input prompts
        - base_answer / source_answer: Expected completions
        - entity: The entity (e.g., city name)
        - attribute: The attribute being tested
        """
        if "base_input" in raw_example:
            return raw_example

        return {
            "base_input": raw_example.get("base_prompt", raw_example.get("input", "")),
            "source_input": raw_example.get("source_prompt", raw_example.get("source_input", "")),
            "base_answer": raw_example.get("base_answer", raw_example.get("label", "")),
            "source_answer": raw_example.get("source_answer", raw_example.get("inv_label", "")),
            "intervention_position": raw_example.get("intervention_position", -2),
            "metadata": {
                "entity": raw_example.get("entity", ""),
                "attribute": raw_example.get("attribute", self.attribute),
                "task_type": "ravel",
            },
        }
