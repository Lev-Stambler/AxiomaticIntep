"""AI2 Reasoning Challenge (ARC) task for MIB benchmark."""

from __future__ import annotations

from .base import BaseMIBTask


class ARCTask(BaseMIBTask):
    """
    AI2 Reasoning Challenge task.

    Tests model reasoning on science questions from the ARC dataset.
    Available in two difficulty levels: Easy and Challenge.

    Example:
    Q: "Which property of air does a barometer measure?
        A) speed B) pressure C) humidity D) temperature"
    Expected: "B"
    """

    def __init__(
        self,
        model,
        layer: int,
        device: str = "cuda",
        difficulty: str = "easy",
    ):
        """
        Initialize ARC task.

        Args:
            model: HookedTransformer model
            layer: Layer index for interventions
            device: Device to run on
            difficulty: "easy" or "challenge"
        """
        super().__init__(model, layer, device)
        self.difficulty = difficulty

    def get_token_position(self, example: dict) -> int:
        """Intervene at reasoning position."""
        return example.get("intervention_position", -2)

    def get_expected_output(self, example: dict, intervened: bool) -> str:
        """Return answer based on intervention."""
        if intervened:
            return example["source_answer"]
        return example["base_answer"]

    def process_example(self, raw_example: dict) -> dict:
        """
        Convert ARC dataset example to standard format.

        ARC examples have:
        - question: The science question
        - choices: List of answer choices
        - answerKey: The correct answer (A, B, C, D)
        """
        if "base_input" in raw_example:
            return raw_example

        # Build formatted question
        question = raw_example.get("question", "")
        choices = raw_example.get("choices", {})

        # Handle different choice formats
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choice_str = " ".join(
                f"{label}) {text}" for label, text in zip(labels, texts, strict=False)
            )
        elif isinstance(choices, list):
            choice_str = " ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
        else:
            choice_str = ""

        formatted_question = f"{question} {choice_str}".strip()

        # For ARC, we typically don't have counterfactual pairs
        # Use same question but track the answer
        return {
            "base_input": formatted_question,
            "source_input": raw_example.get("source_input", formatted_question),
            "base_answer": raw_example.get("base_answer", raw_example.get("answerKey", "")),
            "source_answer": raw_example.get(
                "source_answer", raw_example.get("counterfactual_answerKey", "")
            ),
            "intervention_position": raw_example.get("intervention_position", -2),
            "metadata": {
                "question_id": raw_example.get("id", ""),
                "difficulty": self.difficulty,
                "task_type": f"arc_{self.difficulty}",
            },
        }

    def _check_match(self, predicted: str, expected: str) -> float:
        """Check if answer choice matches."""
        pred_clean = predicted.strip().upper()
        exp_clean = expected.strip().upper()

        # Extract letter if present
        for char in pred_clean:
            if char in "ABCDE":
                pred_clean = char
                break

        for char in exp_clean:
            if char in "ABCDE":
                exp_clean = char
                break

        return 1.0 if pred_clean == exp_clean else 0.0
