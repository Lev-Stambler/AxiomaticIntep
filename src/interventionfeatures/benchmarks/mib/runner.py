"""MIB (Mechanistic Interpretability Benchmark) runner.

Wraps the official MIB implementation: https://github.com/aaronmueller/mib
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn

# Add MIB to path
MIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "interventionfeatures",
    "benchmarks",
    "external",
    "MIB",
    "MIB-causal-variable-track",
)
if MIB_PATH not in sys.path:
    sys.path.append(MIB_PATH)

# Import MIB components
# Fail loudly if MIB is not set up correctly
from CausalAbstraction.neural.pipeline import LMPipeline  # noqa: E402

from ..base import BaseBenchmarkRunner, BenchmarkResult  # noqa: E402


class CSSFeaturizerModule(nn.Module):
    """Adapter to make CSSFeaturizer compatible with MIB's featurizer module interface."""
    def __init__(self, css_featurizer):
        super().__init__()
        self.css = css_featurizer

    def forward(self, x):
        # x -> (features, error)
        # We pass x itself as 'error' (residual context) because
        # CSSFeaturizer.inverse_featurizer expects the full original activation
        # to compute the orthogonal component.
        f = self.css.forward_featurizer(x)
        return f, x


class CSSInverseFeaturizerModule(nn.Module):
    """Adapter to make CSSFeaturizer compatible with MIB's inverse featurizer interface."""
    def __init__(self, css_featurizer):
        super().__init__()
        self.css = css_featurizer

    def forward(self, f, error):
        # (features, error) -> reconstructed x
        # error is the original x (see CSSFeaturizerModule)
        return self.css.inverse_featurizer(f, error)


class MIBBenchmarkRunner(BaseBenchmarkRunner):
    """
    Runner for MIB Causal Variable Track using official implementation.
    """

    # Task name mapping to MIB module names
    TASK_MAPPING = {
        "ioi": "IOI_task",
        "arithmetic": "two_digit_addition_task", # Mapped from 'arithmetic'
        "mcqa": "simple_MCQA",
        "arc_easy": "ARC",
        "ravel": "RAVEL",
    }

    def __init__(
        self,
        css_directions: list[dict],
        model_name: str,
        layer: int,
        device: str = "cuda",
        hf_cache_dir: str | None = None,
    ):
        super().__init__(css_directions, model_name, layer, device)
        self.hf_cache_dir = hf_cache_dir
        # Note: We don't load self._model here because MIB uses its own LMPipeline

    def get_available_tasks(self) -> list[str]:
        return list(self.TASK_MAPPING.keys())

    def load_data(self, task_name: str) -> Any:
        # MIB handles data loading internally in get_counterfactual_datasets
        return None

    def _get_task_modules(self, task_name: str):
        """Import task-specific modules dynamically."""
        # Note: 'raw_output' is the standard output variable name in MIB causal models
        variable = "raw_output"

        if task_name == "ioi":
            from tasks.IOI_task.ioi_task import (
                get_causal_model,
                get_counterfactual_datasets,
                get_token_positions,
            )
        elif task_name == "arithmetic":
            from tasks.two_digit_addition_task.arithmetic import (
                get_causal_model,
                get_counterfactual_datasets,
                get_token_positions,
            )
        elif task_name == "mcqa":
            from tasks.simple_MCQA.simple_MCQA import (
                get_causal_model,
                get_counterfactual_datasets,
                get_token_positions,
            )
        elif task_name == "arc_easy":
            from tasks.ARC.ARC import (
                get_causal_model,
                get_counterfactual_datasets,
                get_token_positions,
            )
        elif task_name == "ravel":
            from tasks.RAVEL.ravel import (
                get_causal_model,
                get_counterfactual_datasets,
                get_token_positions,
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")

        return get_counterfactual_datasets, get_causal_model, get_token_positions, variable

    def _get_checker(self, task_name: str):
        import re

        def simple_checker(output_text, expected):
            return expected in output_text

        def arithmetic_checker(output_text, expected):
            numbers_in_output = re.findall(r'\d+', output_text)
            if not numbers_in_output:
                return False
            first_number = numbers_in_output[0]
            if expected[0] == "0":
                expected_no_leading_zero = expected[1:]
                return first_number == expected_no_leading_zero or first_number == expected
            return first_number == expected

        def ravel_checker(output_text, expected):
            if output_text is None:
                return False
            output_clean = re.sub(r'[^\w\s]+', '', output_text.lower()).strip()
            expected_list = [e.strip().lower() for e in expected.split(',')]
            if any(part in output_clean for part in expected_list):
                return True
            return False

        if task_name == "arithmetic":
            return arithmetic_checker
        elif task_name == "ravel":
            return ravel_checker
        return simple_checker

    def _filter_examples(
        self,
        examples: list,
        pipeline,
        model,
        checker,
        variable: str,
    ) -> list:
        """
        Filter examples to only those where the model performs correctly.

        Following the canonical MIB approach, we only evaluate on examples where
        the model can already produce correct outputs on both base and
        counterfactual inputs without intervention.

        Args:
            examples: Labeled examples from the causal model
            pipeline: LMPipeline for tokenization
            model: HookedTransformer for inference
            checker: Function to check correctness
            variable: The causal variable being tested

        Returns:
            Filtered list of examples
        """
        filtered = []

        for example in examples:
            base_input = example["input"]
            base_label = example.get("label", "")

            # Check if model gets base input correct
            base_tokens = pipeline.load(base_input)
            with torch.no_grad():
                output_ids = model.generate(
                    base_tokens["input_ids"],
                    max_new_tokens=pipeline.max_new_tokens,
                    do_sample=False,
                    eos_token_id=pipeline.tokenizer.eos_token_id,
                    verbose=False
                )
            new_tokens = output_ids[0, base_tokens["input_ids"].shape[1]:]
            base_pred = pipeline.tokenizer.decode(new_tokens, skip_special_tokens=True)

            if not checker(base_pred, base_label):
                continue  # Model fails on base input

            # Check if model gets counterfactual input correct
            # (counterfactual label is stored in 'counterfactual_labels')
            source_input = example["counterfactual_inputs"][0]
            cf_labels = example.get("counterfactual_labels", {})
            cf_label = cf_labels.get(variable, [""])[0] if cf_labels else ""

            if cf_label:
                source_tokens = pipeline.load(source_input)
                with torch.no_grad():
                    output_ids = model.generate(
                        source_tokens["input_ids"],
                        max_new_tokens=pipeline.max_new_tokens,
                        do_sample=False,
                        eos_token_id=pipeline.tokenizer.eos_token_id,
                        verbose=False
                    )
                new_tokens = output_ids[0, source_tokens["input_ids"].shape[1]:]
                source_pred = pipeline.tokenizer.decode(new_tokens, skip_special_tokens=True)

                if not checker(source_pred, cf_label):
                    continue  # Model fails on counterfactual input

            filtered.append(example)

        return filtered

    def run_evaluation(
        self,
        task_name: str,
        direction_indices: list[int] | None = None,
        num_samples: int | None = None,
        filter_examples: bool = True,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run MIB evaluation with manual intervention loop.

        Args:
            task_name: MIB task to evaluate
            direction_indices: Which CSS directions to use
            num_samples: Limit number of samples
            filter_examples: Whether to filter to examples where model performs correctly.
                           This follows the canonical MIB approach for fair evaluation.
        """
        # 1. Setup task components
        get_cf_datasets, get_causal_model, get_token_pos, variable = self._get_task_modules(task_name)

        # 2. Setup Pipeline (for data loading and tokenization helpers)
        # We also need self._model (HookedTransformer) for interventions
        if self._model is None:
            self._load_model()

        pipeline = LMPipeline(self.model_name, max_new_tokens=10, device=self.device)
        pipeline.tokenizer.padding_side = "left"

        # Use our HookedTransformer for interventions
        model = self._model

        # 3. Get Data
        dataset_size = num_samples if num_samples else None
        try:
            cf_datasets = get_cf_datasets(hf=True, size=dataset_size)
        except TypeError:
            try:
                cf_datasets = get_cf_datasets(size=dataset_size)
            except Exception:
                cf_datasets = get_cf_datasets()

        cf_datasets = {k: v for k, v in cf_datasets.items() if "test" in k}
        if not cf_datasets:
            return BenchmarkResult(task_name, {"iia": 0.0}, [], {"error": "No datasets found"})

        # 4. Setup Causal Model and Token Positions
        import inspect
        sig = inspect.signature(get_causal_model)
        if len(sig.parameters) > 0:
            params = {"position_coeff": 0, "token_coeff": 0, "bias": 0}
            causal_model = get_causal_model(params)
        else:
            causal_model = get_causal_model()

        token_positions = get_token_pos(pipeline, causal_model)

        # 5. Create Featurizer
        css_featurizer = self.get_featurizer(direction_indices)
        css_featurizer = css_featurizer.to(self.device)

        # 6. Checker function
        checker = self._get_checker(task_name)
        hook_name = f"blocks.{self.layer}.hook_resid_post"

        all_scores = []
        total_filtered = 0
        total_original = 0
        from tqdm import tqdm

        for ds_name, dataset in cf_datasets.items():
            print(f"  Evaluating dataset: {ds_name}")
            ds_scores = []

            # Label the dataset using the causal model to get expected labels for interventions
            labeled_examples = causal_model.label_counterfactual_data(dataset, [variable])
            total_original += len(labeled_examples)

            # 7. Filter examples (canonical MIB approach)
            if filter_examples:
                print("    Filtering to examples where model performs correctly...")
                labeled_examples = self._filter_examples(
                    labeled_examples, pipeline, model, checker, variable
                )
                print(f"    Kept {len(labeled_examples)} examples after filtering")

            total_filtered += len(labeled_examples)

            if not labeled_examples:
                print("    No examples passed filtering, skipping dataset")
                continue

            # 8. Manual Intervention Loop
            for i in tqdm(range(len(labeled_examples)), desc=f"    {task_name}"):
                example = labeled_examples[i]
                base_input = example["input"]
                # MIB counterfactual_inputs is a list of lists
                source_input = example["counterfactual_inputs"][0]

                # For IIA, we want to check if intervention produces the SOURCE label
                # (i.e., does swapping the representation make output match counterfactual?)
                cf_labels = example.get("counterfactual_labels", {})
                expected_label = cf_labels.get(variable, [""])[0] if cf_labels else ""

                # Fallback to base label if no counterfactual label available
                if not expected_label:
                    expected_label = example.get("label", "")

                # Tokenize
                base_tokens = pipeline.load(base_input)
                source_tokens = pipeline.load(source_input)

                # Get intervention positions for THIS example
                # Note: For IOI, the indexer expects the RAW input (base_input), not tokenized
                pos_indices = token_positions[0].index(base_input)
                if isinstance(pos_indices, list):
                    pos = pos_indices[0]
                else:
                    pos = pos_indices

                # Get activations
                with torch.no_grad():
                    _, base_cache = model.run_with_cache(base_tokens["input_ids"], names_filter=[hook_name])
                    _, source_cache = model.run_with_cache(source_tokens["input_ids"], names_filter=[hook_name])

                base_act = base_cache[hook_name][:, pos, :]
                source_act = source_cache[hook_name][:, pos, :]

                # Apply CSS Featurizer (interchange intervention)
                intervened_act = css_featurizer(base_act, source_act)

                # Run model with intervention
                # We only want to apply the intervention during the prompt processing
                # (first forward pass). During generation, sequence length will be 1.
                prompt_len = base_tokens["input_ids"].shape[1]

                def make_intervention_hook(p_len, p, i_act):
                    def intervention_hook(activation, hook):
                        # Only apply intervention if we are processing the prompt
                        if activation.shape[1] == p_len:
                            activation[:, p, :] = i_act
                        return activation
                    return intervention_hook

                with model.hooks(fwd_hooks=[(hook_name, make_intervention_hook(prompt_len, pos, intervened_act))]):
                    output_ids = model.generate(
                        base_tokens["input_ids"],
                        max_new_tokens=pipeline.max_new_tokens,
                        do_sample=False,
                        eos_token_id=pipeline.tokenizer.eos_token_id,
                        verbose=False
                    )

                # Decode output (excluding prompt)
                new_tokens = output_ids[0, base_tokens["input_ids"].shape[1]:]
                predicted_text = pipeline.tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Check match - IIA measures if intervention produces the counterfactual output
                is_correct = checker(predicted_text, expected_label)
                ds_scores.append(float(is_correct))

            if ds_scores:
                all_scores.append(sum(ds_scores) / len(ds_scores))

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return BenchmarkResult(
            task_name=task_name,
            metrics={"iia": avg_score},
            metadata={
                "num_datasets": len(cf_datasets),
                "variable": variable,
                "filtered": filter_examples,
                "examples_after_filter": total_filtered,
                "examples_before_filter": total_original,
            }
        )

    def run_all_evaluations(
        self,
        direction_indices: list[int] | None = None,
        tasks: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> dict[str, BenchmarkResult]:
        """
        Run all MIB tasks.

        Args:
            direction_indices: Which CSS directions to use
            tasks: Specific tasks to run, None = all available
            num_samples: Limit samples per task

        Returns:
            Dictionary mapping task names to results
        """
        if tasks is None:
            tasks = self.get_available_tasks()

        # Filter to only valid tasks
        valid_tasks = []
        for task in tasks:
            if task in self.TASK_MAPPING:
                valid_tasks.append(task)
            else:
                print(f"Warning: Unknown MIB task '{task}', skipping")

        results = {}
        for task_name in valid_tasks:
            print(f"\nEvaluating task: {task_name}")
            try:
                results[task_name] = self.run_evaluation(
                    task_name,
                    direction_indices,
                    num_samples=num_samples,
                    **kwargs
                )
            except Exception as e:
                print(f"Error evaluating {task_name}: {e}")
                import traceback
                traceback.print_exc()
                results[task_name] = BenchmarkResult(
                    task_name=task_name,
                    metrics={"iia": 0.0},
                    metadata={"error": str(e)}
                )

        return results

    def get_aggregate_score(self, results: dict[str, BenchmarkResult]) -> float:
        """
        Compute aggregate IIA score across all tasks.

        Args:
            results: Dictionary of task_name -> BenchmarkResult

        Returns:
            Mean IIA across all tasks
        """
        iia_scores = []
        for result in results.values():
            iia = result.metrics.get("iia", 0.0)
            iia_scores.append(iia)

        return sum(iia_scores) / len(iia_scores) if iia_scores else 0.0
