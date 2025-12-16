"""Multiple Choice Question Answering (MCQA) task for MIB benchmark."""

from __future__ import annotations

from .base import BaseMIBTask


class MCQATask(BaseMIBTask):
    """
    Multiple Choice Question Answering task.

    Tests whether the model can answer multiple choice questions about
    object properties (like colors) and whether swapping the object
    representation causes different answer selection.

    Example (CopyColors MCQA):
    Base: "The apple is red. What color is the apple? A) red B) blue" -> "A"
    Source: "The apple is blue. What color is the apple? A) red B) blue" -> "B"
    """

    def get_token_position(self, example: dict) -> int:
        """Intervene at position where object property is encoded."""
        return example.get("intervention_position", -2)

    def get_expected_output(self, example: dict, intervened: bool) -> str:
        """Return answer choice based on intervention."""
        if intervened:
            return example["source_answer"]
        return example["base_answer"]

    def process_example(self, raw_example: dict) -> dict:
        """
        Convert MCQA dataset example to standard format.

        MCQA examples typically have:
        - question: The question text with choices
        - answer: The correct choice (A, B, C, D)
        - context: Background information
        """
        if "base_input" in raw_example:
            return raw_example

        # Build input from question and choices if needed
        base_context = raw_example.get("base_context", raw_example.get("context", ""))
        source_context = raw_example.get(
            "source_context", raw_example.get("counterfactual_context", "")
        )

        question = raw_example.get("question", "")
        choices = raw_example.get("choices", [])

        # Format choices if available
        if choices:
            choice_str = " ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
            question = f"{question} {choice_str}"

        base_input = f"{base_context} {question}".strip()
        source_input = f"{source_context} {question}".strip()

        return {
            "base_input": base_input,
            "source_input": source_input,
            "base_answer": raw_example.get("base_answer", raw_example.get("answer", "")),
            "source_answer": raw_example.get(
                "source_answer", raw_example.get("counterfactual_answer", "")
            ),
            "intervention_position": raw_example.get("intervention_position", -2),
            "metadata": {
                "question": question,
                "choices": choices,
                "task_type": "mcqa",
            },
        }

    def _check_match(self, predicted: str, expected: str) -> float:
        """Check if answer choice matches."""
        pred_clean = predicted.strip().upper()
        exp_clean = expected.strip().upper()

        # Extract letter if present
        for char in pred_clean:
            if char in "ABCD":
                pred_clean = char
                break

        for char in exp_clean:
            if char in "ABCD":
                exp_clean = char
                break

        return 1.0 if pred_clean == exp_clean else 0.0
