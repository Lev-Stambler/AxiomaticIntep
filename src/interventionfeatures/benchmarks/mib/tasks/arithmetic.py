"""Arithmetic task for MIB benchmark."""

from __future__ import annotations

from .base import BaseMIBTask


class ArithmeticTask(BaseMIBTask):
    """
    Arithmetic task (addition and subtraction).

    Tests whether the model can perform two-digit arithmetic and whether
    swapping operand representations causes the model to compute with
    the source's operands instead.

    Example:
    Base: "23 + 45 ="  -> Expected: "68"
    Source: "23 + 12 =" -> Expected: "35"
    After intervention (swap second operand): "23 + 12 =" -> "35"
    """

    def __init__(self, model, layer: int, device: str = "cuda", operation: str = "add"):
        """
        Initialize arithmetic task.

        Args:
            model: HookedTransformer model
            layer: Layer index for interventions
            device: Device to run on
            operation: "add" or "sub" for addition/subtraction
        """
        super().__init__(model, layer, device)
        self.operation = operation

    def get_token_position(self, example: dict) -> int:
        """Intervene at the operand position."""
        # Typically intervene at the position of the second operand
        return example.get("intervention_position", -2)

    def get_expected_output(self, example: dict, intervened: bool) -> str:
        """Return result based on intervention."""
        if intervened:
            return example["source_answer"]
        return example["base_answer"]

    def process_example(self, raw_example: dict) -> dict:
        """
        Convert arithmetic dataset example to standard format.

        Arithmetic examples typically have:
        - equation: The arithmetic expression
        - answer: The correct result
        - operand1, operand2: The operands
        """
        if "base_input" in raw_example:
            return raw_example

        # Convert from MIB format
        base_eq = raw_example.get("base_equation", raw_example.get("equation", ""))
        source_eq = raw_example.get(
            "source_equation", raw_example.get("counterfactual_equation", "")
        )

        return {
            "base_input": base_eq,
            "source_input": source_eq,
            "base_answer": str(raw_example.get("base_answer", raw_example.get("answer", ""))),
            "source_answer": str(
                raw_example.get("source_answer", raw_example.get("counterfactual_answer", ""))
            ),
            "intervention_position": raw_example.get("intervention_position", -2),
            "metadata": {
                "operation": self.operation,
                "operand1": raw_example.get("operand1", ""),
                "operand2": raw_example.get("operand2", ""),
                "task_type": f"arithmetic_{self.operation}",
            },
        }

    def _check_match(self, predicted: str, expected: str) -> float:
        """Check if numeric prediction matches expected."""
        pred_clean = predicted.strip()
        exp_clean = expected.strip()

        # Try numeric comparison
        try:
            pred_num = int("".join(c for c in pred_clean if c.isdigit() or c == "-"))
            exp_num = int(exp_clean)
            return 1.0 if pred_num == exp_num else 0.0
        except (ValueError, AttributeError):
            pass

        # Fall back to string comparison
        return super()._check_match(predicted, expected)
