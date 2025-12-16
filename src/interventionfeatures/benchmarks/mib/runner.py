"""MIB (Mechanistic Interpretability Benchmark) runner.

MIB evaluates featurization methods on their ability to localize causal variables
using Interchange Intervention Accuracy (IIA).

Reference: https://github.com/aaronmueller/mib
Paper: https://arxiv.org/abs/2504.13151
"""

from __future__ import annotations

from ..base import BaseBenchmarkRunner, BenchmarkResult
from .tasks import (
    ARCTask,
    ArithmeticTask,
    BaseMIBTask,
    IOITask,
    MCQATask,
    RAVELTask,
)


class MIBBenchmarkRunner(BaseBenchmarkRunner):
    """
    Runner for MIB Causal Variable Track.

    Implements evaluation on all 5 MIB tasks:
    - IOI (Indirect Object Identification)
    - Arithmetic (Addition/Subtraction)
    - MCQA (Multiple Choice QA)
    - ARC (AI2 Reasoning Challenge - Easy and Challenge)
    - RAVEL

    Uses Interchange Intervention Accuracy (IIA) as the primary metric.
    """

    # Task name to class mapping
    TASK_CLASSES: dict[str, type[BaseMIBTask]] = {
        "ioi": IOITask,
        "arithmetic_add": ArithmeticTask,
        "arithmetic_sub": ArithmeticTask,
        "mcqa": MCQATask,
        "arc_easy": ARCTask,
        "arc_challenge": ARCTask,
        "ravel": RAVELTask,
    }

    # HuggingFace dataset paths
    HF_DATASETS = {
        "ioi": "mib-bench/ioi",
        "arithmetic_add": "mib-bench/arithmetic_addition",
        "arithmetic_sub": "mib-bench/arithmetic_subtraction",
        "mcqa": "mib-bench/copycolors_mcqa",
        "arc_easy": "mib-bench/arc_easy",
        "arc_challenge": "mib-bench/arc_challenge",
        "ravel": "mib-bench/ravel",
    }

    def __init__(
        self,
        css_directions: list[dict],
        model_name: str,
        layer: int,
        device: str = "cuda",
        hf_cache_dir: str | None = None,
    ):
        """
        Initialize MIB benchmark runner.

        Args:
            css_directions: Output from CSSDirectionFinder
            model_name: HuggingFace model name
            layer: Layer index for interventions
            device: Device to run on
            hf_cache_dir: Optional HuggingFace cache directory
        """
        super().__init__(css_directions, model_name, layer, device)
        self.hf_cache_dir = hf_cache_dir
        self._task_instances: dict[str, BaseMIBTask] = {}

    def get_available_tasks(self) -> list[str]:
        """Return available MIB tasks."""
        return list(self.TASK_CLASSES.keys())

    def _get_task_instance(self, task_name: str) -> BaseMIBTask:
        """Get or create task instance."""
        if task_name not in self._task_instances:
            if self._model is None:
                self._load_model()

            task_class = self.TASK_CLASSES.get(task_name)
            if not task_class:
                raise ValueError(f"Unknown task: {task_name}")

            # Create task with appropriate parameters
            kwargs = {
                "model": self._model,
                "layer": self.layer,
                "device": self.device,
            }

            # Add task-specific parameters
            if task_name == "arithmetic_add":
                kwargs["operation"] = "add"
            elif task_name == "arithmetic_sub":
                kwargs["operation"] = "sub"
            elif task_name == "arc_easy":
                kwargs["difficulty"] = "easy"
            elif task_name == "arc_challenge":
                kwargs["difficulty"] = "challenge"

            self._task_instances[task_name] = task_class(**kwargs)

        return self._task_instances[task_name]

    def load_data(self, task_name: str) -> list[dict]:
        """
        Load MIB dataset from HuggingFace.

        Args:
            task_name: Name of the task

        Returns:
            List of processed examples
        """
        dataset_name = self.HF_DATASETS.get(task_name)
        if not dataset_name:
            raise ValueError(f"Unknown task: {task_name}")

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                dataset_name,
                split="test",
                cache_dir=self.hf_cache_dir,
            )

            # Get task instance for processing
            task = self._get_task_instance(task_name)

            # Process each example
            processed = []
            for item in dataset:
                processed.append(task.process_example(dict(item)))

            return processed

        except Exception as e:
            print(f"Warning: Could not load {dataset_name} from HuggingFace: {e}")
            return self._generate_synthetic_data(task_name)

    def _generate_synthetic_data(self, task_name: str) -> list[dict]:
        """Generate synthetic test data for demonstration."""
        examples = []

        if task_name == "ioi":
            # IOI synthetic examples
            pairs = [
                ("John gave Mary the book. Mary gave", "John", "Mary"),
                ("Alice sent Bob a letter. Bob sent", "Alice", "Bob"),
                ("Tom helped Sarah with homework. Sarah helped", "Tom", "Sarah"),
            ]
            for base, base_io, source_io in pairs:
                examples.append(
                    {
                        "base_input": f"{base} the book to",
                        "source_input": base.replace(base_io, source_io) + " the book to",
                        "base_answer": base_io,
                        "source_answer": source_io,
                        "intervention_position": -2,
                        "metadata": {"task_type": "ioi"},
                    }
                )

        elif task_name.startswith("arithmetic"):
            op = "+" if "add" in task_name else "-"
            for a in [12, 23, 34]:
                for b in [11, 22, 33]:
                    result = a + b if op == "+" else a - b
                    alt_b = b + 10
                    alt_result = a + alt_b if op == "+" else a - alt_b
                    examples.append(
                        {
                            "base_input": f"{a} {op} {b} =",
                            "source_input": f"{a} {op} {alt_b} =",
                            "base_answer": str(result),
                            "source_answer": str(alt_result),
                            "intervention_position": -2,
                            "metadata": {"task_type": task_name},
                        }
                    )

        elif task_name == "mcqa":
            qa_pairs = [
                ("The sky is blue.", "blue", "What color is the sky?", ["blue", "red"]),
                ("The grass is green.", "green", "What color is the grass?", ["green", "yellow"]),
            ]
            for context, answer, question, choices in qa_pairs:
                choice_str = " ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
                examples.append(
                    {
                        "base_input": f"{context} {question} {choice_str}",
                        "source_input": f"{context.replace(answer, choices[1])} {question} {choice_str}",
                        "base_answer": "A",
                        "source_answer": "B",
                        "intervention_position": -2,
                        "metadata": {"task_type": "mcqa"},
                    }
                )

        elif task_name.startswith("arc"):
            examples.append(
                {
                    "base_input": "What gas do plants absorb? A) oxygen B) carbon dioxide C) nitrogen",
                    "source_input": "What gas do plants absorb? A) oxygen B) carbon dioxide C) nitrogen",
                    "base_answer": "B",
                    "source_answer": "B",
                    "intervention_position": -2,
                    "metadata": {"task_type": task_name},
                }
            )

        elif task_name == "ravel":
            cities = [
                ("Paris", "France", "Berlin", "Germany"),
                ("Tokyo", "Japan", "Seoul", "South Korea"),
            ]
            for base_city, base_country, src_city, src_country in cities:
                examples.append(
                    {
                        "base_input": f"{base_city} is located in",
                        "source_input": f"{src_city} is located in",
                        "base_answer": base_country,
                        "source_answer": src_country,
                        "intervention_position": -2,
                        "metadata": {
                            "entity": base_city,
                            "attribute": "country",
                            "task_type": "ravel",
                        },
                    }
                )

        return examples

    def run_evaluation(
        self,
        task_name: str,
        direction_indices: list[int] | None = None,
        num_samples: int | None = None,
    ) -> BenchmarkResult:
        """
        Run MIB evaluation for a specific task.

        Args:
            task_name: One of the available task names
            direction_indices: Which CSS directions to use
            num_samples: Limit evaluation to N samples

        Returns:
            BenchmarkResult with IIA score
        """
        # Ensure model is loaded
        if self._model is None:
            self._load_model()

        # Get task instance
        task = self._get_task_instance(task_name)

        # Load data
        examples = self.load_data(task_name)

        if num_samples and num_samples > 0:
            examples = examples[:num_samples]

        if not examples:
            return BenchmarkResult(
                task_name=task_name,
                metrics={"iia": 0.0},
                per_sample_results=[],
                metadata={"error": "No examples loaded"},
            )

        # Create featurizer
        featurizer = self.get_featurizer(direction_indices)
        featurizer = featurizer.to(self.device)

        # Run evaluation
        iia_results = []
        per_sample_results = []

        for i, example in enumerate(examples):
            try:
                iia = task.evaluate_single(example, featurizer)
                iia_results.append(iia)
                per_sample_results.append(
                    {
                        "index": i,
                        "iia": iia,
                        "base_input": example.get("base_input", "")[:50],
                    }
                )
            except Exception as e:
                print(f"Warning: Error evaluating example {i}: {e}")
                continue

        # Compute aggregate metric
        iia_score = sum(iia_results) / len(iia_results) if iia_results else 0.0

        return BenchmarkResult(
            task_name=task_name,
            metrics={"iia": iia_score},
            per_sample_results=per_sample_results,
            metadata={
                "num_samples": len(examples),
                "num_evaluated": len(iia_results),
                "model": self.model_name,
                "layer": self.layer,
                "num_directions": featurizer.num_directions,
            },
        )

    def run_all_evaluations(
        self,
        direction_indices: list[int] | None = None,
        tasks: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> dict[str, BenchmarkResult]:
        """
        Run evaluation on multiple MIB tasks.

        Args:
            direction_indices: Which CSS directions to use
            tasks: Specific tasks to run, None = all
            num_samples: Limit samples per task

        Returns:
            Dictionary mapping task names to results
        """
        if tasks is None:
            tasks = self.get_available_tasks()

        results = {}
        for task_name in tasks:
            if task_name not in self.TASK_CLASSES:
                print(f"Warning: Unknown task {task_name}, skipping")
                continue

            print(f"Running MIB task: {task_name}")
            results[task_name] = self.run_evaluation(task_name, direction_indices, num_samples)
            print(f"  IIA: {results[task_name].metrics['iia']:.4f}")

        return results

    def get_aggregate_score(self, results: dict[str, BenchmarkResult]) -> float:
        """
        Compute aggregate IIA score across all tasks.

        Args:
            results: Dictionary of task results

        Returns:
            Mean IIA across all tasks
        """
        if not results:
            return 0.0

        scores = [r.metrics.get("iia", 0.0) for r in results.values()]
        return sum(scores) / len(scores)
