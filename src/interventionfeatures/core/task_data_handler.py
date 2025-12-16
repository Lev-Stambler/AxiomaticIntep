"""Task data handler for loading benchmark task datasets for targeted CSS discovery."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Task dataset mappings
TASK_DATASET_MAP = {
    # IOI task
    "ioi": "mib-bench/ioi",
    # Arithmetic tasks
    "arithmetic_add": "mib-bench/arithmetic_addition",
    "arithmetic_sub": "mib-bench/arithmetic_subtraction",
    # MCQA task
    "mcqa": "mib-bench/copycolors_mcqa",
    # ARC tasks
    "arc_easy": "mib-bench/arc_easy",
    "arc_challenge": "mib-bench/arc_challenge",
    # RAVEL tasks (by attribute)
    "ravel_country": "mib-bench/ravel",
    "ravel_language": "mib-bench/ravel",
    "ravel_continent": "mib-bench/ravel",
    "ravel_timezone": "mib-bench/ravel",
}


class TaskDataHandler:
    """
    Load and process benchmark task datasets for targeted CSS discovery.

    Provides contrastive batches of (base_input, source_input, expected_output)
    for task-targeted optimization.
    """

    def __init__(
        self,
        task_name: str,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        max_seq_len: int = 128,
        hf_cache_dir: str | None = None,
    ):
        """
        Initialize task data handler.

        Args:
            task_name: Name of the task (e.g., "ioi", "ravel_country")
            tokenizer: Tokenizer for encoding inputs
            device: Device for tensors
            max_seq_len: Maximum sequence length for tokenization
            hf_cache_dir: Optional HuggingFace cache directory
        """
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.hf_cache_dir = hf_cache_dir

        # Load and process dataset
        self.dataset = self._load_task_dataset()
        self.processed_examples = self._process_all_examples()
        self._example_idx = 0

        print(f"TaskDataHandler initialized for task: {task_name}")
        print(f"  Loaded {len(self.processed_examples)} examples")

    def _load_task_dataset(self):
        """Load dataset from HuggingFace."""
        dataset_path = TASK_DATASET_MAP.get(self.task_name)

        if dataset_path is None:
            raise ValueError(
                f"Unknown task: {self.task_name}. "
                f"Available tasks: {list(TASK_DATASET_MAP.keys())}"
            )

        try:
            dataset = load_dataset(
                dataset_path,
                split="train",
                cache_dir=self.hf_cache_dir,
                token=os.environ.get("HUGGINGFACE_API_KEY"),
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_path}: {e}") from e

    def _process_all_examples(self) -> list[dict]:
        """Process all dataset examples into standard format."""
        processed = []

        for raw_example in self.dataset:
            example = self._process_example(raw_example)
            if example is not None:
                processed.append(example)

        return processed

    def _process_example(self, raw_example: dict) -> dict | None:
        """
        Process a single example into standard format.

        Returns:
            Dictionary with:
                - base_input: str
                - source_input: str
                - base_answer: str
                - source_answer: str
                - intervention_position: int
            Or None if the example should be skipped.
        """
        if self.task_name == "ioi":
            return self._process_ioi_example(raw_example)
        elif self.task_name.startswith("arithmetic"):
            return self._process_arithmetic_example(raw_example)
        elif self.task_name.startswith("ravel"):
            return self._process_ravel_example(raw_example)
        elif self.task_name == "mcqa":
            return self._process_mcqa_example(raw_example)
        elif self.task_name.startswith("arc"):
            return self._process_arc_example(raw_example)
        else:
            # Generic fallback
            return self._process_generic_example(raw_example)

    def _process_ioi_example(self, raw: dict) -> dict:
        """Process IOI example."""
        return {
            "base_input": raw.get("base_text", raw.get("text", "")),
            "source_input": raw.get("source_text", raw.get("counterfactual_text", "")),
            "base_answer": raw.get("base_io", raw.get("IO", "")),
            "source_answer": raw.get("source_io", raw.get("counterfactual_IO", "")),
            "intervention_position": raw.get("intervention_position", -2),
        }

    def _process_arithmetic_example(self, raw: dict) -> dict:
        """Process arithmetic example."""
        return {
            "base_input": raw.get("base_text", raw.get("input", "")),
            "source_input": raw.get("source_text", raw.get("counterfactual_input", "")),
            "base_answer": str(raw.get("base_answer", raw.get("answer", ""))),
            "source_answer": str(raw.get("source_answer", raw.get("counterfactual_answer", ""))),
            "intervention_position": raw.get("intervention_position", -1),
        }

    def _process_ravel_example(self, raw: dict) -> dict | None:
        """Process RAVEL example, filtering by attribute if needed."""
        # Extract attribute from task name (e.g., "ravel_country" -> "country")
        target_attribute = self.task_name.split("_", 1)[1] if "_" in self.task_name else None

        # Filter by attribute if specified
        if target_attribute and raw.get("attribute") != target_attribute:
            return None

        return {
            "base_input": raw.get("base_prompt", raw.get("input", "")),
            "source_input": raw.get("source_prompt", raw.get("source_input", "")),
            "base_answer": raw.get("base_answer", raw.get("label", "")),
            "source_answer": raw.get("source_answer", raw.get("inv_label", "")),
            "intervention_position": raw.get("intervention_position", -2),
        }

    def _process_mcqa_example(self, raw: dict) -> dict:
        """Process MCQA example."""
        return {
            "base_input": raw.get("base_text", raw.get("question", "")),
            "source_input": raw.get("source_text", raw.get("counterfactual_question", "")),
            "base_answer": raw.get("base_answer", raw.get("answer", "")),
            "source_answer": raw.get("source_answer", raw.get("counterfactual_answer", "")),
            "intervention_position": raw.get("intervention_position", -1),
        }

    def _process_arc_example(self, raw: dict) -> dict:
        """Process ARC example."""
        return {
            "base_input": raw.get("base_text", raw.get("question", "")),
            "source_input": raw.get("source_text", raw.get("counterfactual_question", "")),
            "base_answer": raw.get("base_answer", raw.get("answerKey", "")),
            "source_answer": raw.get("source_answer", raw.get("counterfactual_answerKey", "")),
            "intervention_position": raw.get("intervention_position", -1),
        }

    def _process_generic_example(self, raw: dict) -> dict:
        """Generic example processing."""
        return {
            "base_input": raw.get("base_input", raw.get("input", "")),
            "source_input": raw.get("source_input", ""),
            "base_answer": raw.get("base_answer", raw.get("answer", "")),
            "source_answer": raw.get("source_answer", ""),
            "intervention_position": raw.get("intervention_position", -1),
        }

    def get_contrastive_batch(self, batch_size: int) -> dict:
        """
        Return a batch of contrastive examples for task-targeted optimization.

        Args:
            batch_size: Number of examples in batch

        Returns:
            Dictionary with:
                - base_tokens: Tensor of shape (B, seq_len)
                - source_tokens: Tensor of shape (B, seq_len)
                - base_answer_tokens: Tensor of shape (B,) - answer token IDs
                - source_answer_tokens: Tensor of shape (B,) - target answer token IDs
                - intervention_positions: Tensor of shape (B,) - positions to intervene
        """
        # Sample batch of examples
        if len(self.processed_examples) <= batch_size:
            batch_examples = self.processed_examples.copy()
            random.shuffle(batch_examples)
        else:
            batch_examples = random.sample(self.processed_examples, batch_size)

        # Tokenize inputs
        base_inputs = [ex["base_input"] for ex in batch_examples]
        source_inputs = [ex["source_input"] for ex in batch_examples]

        base_encoded = self.tokenizer(
            base_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

        source_encoded = self.tokenizer(
            source_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

        base_tokens = base_encoded.input_ids.to(self.device)
        source_tokens = source_encoded.input_ids.to(self.device)

        # Get answer tokens (first token of each answer)
        source_answer_tokens = []
        for ex in batch_examples:
            answer_text = ex["source_answer"]
            # Add leading space for tokenizers that need it
            answer_ids = self.tokenizer.encode(" " + answer_text, add_special_tokens=False)
            # Take first token
            source_answer_tokens.append(answer_ids[0] if answer_ids else 0)

        source_answer_tokens = torch.tensor(source_answer_tokens, device=self.device)

        # Get intervention positions
        intervention_positions = []
        for ex in batch_examples:
            pos = ex["intervention_position"]
            seq_len = base_tokens.shape[1]
            # Handle negative indexing
            if pos < 0:
                pos = seq_len + pos
            intervention_positions.append(pos)

        intervention_positions = torch.tensor(intervention_positions, device=self.device)

        return {
            "base_tokens": base_tokens,
            "source_tokens": source_tokens,
            "source_answer_tokens": source_answer_tokens,
            "intervention_positions": intervention_positions,
            "batch_size": len(batch_examples),
        }

    def get_all_examples(self) -> list[dict]:
        """Return all processed examples."""
        return self.processed_examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.processed_examples)

    def __iter__(self):
        """Iterate over examples."""
        self._example_idx = 0
        return self

    def __next__(self):
        """Get next example."""
        if self._example_idx >= len(self.processed_examples):
            raise StopIteration
        example = self.processed_examples[self._example_idx]
        self._example_idx += 1
        return example
