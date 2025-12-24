"""MIB (Mechanistic Interpretability Benchmark) runner.

Wraps the official MIB implementation: https://github.com/aaronmueller/mib
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
from typing import Any

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

from ..base import BaseBenchmarkRunner, BenchmarkResult

# Import MIB components
try:
    from CausalAbstraction.experiments.residual_stream_experiment import PatchResidualStream
    from CausalAbstraction.neural.featurizers import Featurizer
    from CausalAbstraction.neural.pipeline import LMPipeline
    from CausalAbstraction.neural.LM_units import TokenPosition
except ImportError as e:
    print(f"Error importing MIB components: {e}")
    # We might be in a context where MIB is not yet set up
    pass


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
        if task_name == "ioi":
            from tasks.IOI_task.ioi_task import get_counterfactual_datasets, get_causal_model, get_token_positions
            variable = "indirect_object" 
        elif task_name == "arithmetic":
            from tasks.two_digit_addition_task.arithmetic import get_counterfactual_datasets, get_causal_model, get_token_positions
            variable = "sum" # Default variable
        elif task_name == "mcqa":
            from tasks.simple_MCQA.simple_MCQA import get_counterfactual_datasets, get_causal_model, get_token_positions
            variable = "object"
        elif task_name == "arc_easy":
            from tasks.ARC.ARC import get_counterfactual_datasets, get_causal_model, get_token_positions
            variable = "answer"
        elif task_name == "ravel":
            from tasks.RAVEL.ravel import get_counterfactual_datasets, get_causal_model, get_token_positions
            variable = "country" # Default
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

    def run_evaluation(
        self,
        task_name: str,
        direction_indices: list[int] | None = None,
        num_samples: int | None = None,
    ) -> BenchmarkResult:
        """
        Run MIB evaluation using official PatchResidualStream.
        """
        # 1. Setup task components
        get_cf_datasets, get_causal_model, get_token_pos, variable = self._get_task_modules(task_name)
        
        # 2. Setup Pipeline
        # Map model names to what MIB expects or use full path
        pipeline = LMPipeline(self.model_name, max_new_tokens=10, device=self.device) 
        # Note: max_new_tokens=10 to allow for generation checks
        pipeline.tokenizer.padding_side = "left"

        # 3. Get Data
        # MIB loads data via get_counterfactual_datasets
        # We assume size=None loads all, but we can limit if num_samples is set
        # However, MIB logic usually loads all. We can slice later if needed but 
        # FilterExperiment uses batch_size.
        dataset_size = num_samples if num_samples else None
        
        # MIB's get_counterfactual_datasets usually has signature (hf=True, size=...)
        # But some might differ. IOI uses (size=...).
        try:
            cf_datasets = get_cf_datasets(hf=True, size=dataset_size)
        except TypeError:
            # Fallback for IOI which might not have hf arg in some versions or differs
            try:
                cf_datasets = get_cf_datasets(size=dataset_size)
            except:
                cf_datasets = get_cf_datasets()

        # Filter for test sets
        cf_datasets = {k: v for k, v in cf_datasets.items() if "test" in k}
        
        if not cf_datasets:
            return BenchmarkResult(task_name, {"iia": 0.0}, [], {"error": "No datasets found"})

        # 4. Setup Causal Model and Token Positions
        causal_model = get_causal_model()
        token_positions = get_token_pos(pipeline, causal_model)
        
        # 5. Create Featurizer
        css_featurizer = self.get_featurizer(direction_indices)
        css_featurizer = css_featurizer.to(self.device)
        
        # Wrap in MIB Featurizer
        mib_featurizer = Featurizer(
            featurizer=CSSFeaturizerModule(css_featurizer),
            inverse_featurizer=CSSInverseFeaturizerModule(css_featurizer),
            n_features=css_featurizer.num_directions,
            id="css"
        )
        
        # Construct featurizers dict for all layers/positions
        # MIB expects: {(layer, position_id): featurizer}
        # But PatchResidualStream init logic iterates layers and token_positions 
        # and looks up featurizers. If not found, it creates IdentityFeaturizer.
        # We want to apply OUR featurizer at the specific layer we are testing.
        
        featurizers_map = {}
        for pos in token_positions:
            featurizers_map[(self.layer, pos.id)] = mib_featurizer

        # 6. Run Experiment
        checker = self._get_checker(task_name)
        
        # Filter datasets (optional, but good practice in MIB)
        # filter_experiment = FilterExperiment(pipeline, causal_model, checker)
        # filtered_datasets = filter_experiment.filter(cf_datasets, verbose=False)
        # We skip filtering for speed/simplicity and use all test data
        filtered_datasets = cf_datasets

        experiment = PatchResidualStream(
            pipeline=pipeline,
            causal_model=causal_model,
            layers=[self.layer],
            token_positions=token_positions,
            checker=checker,
            featurizers=featurizers_map,
            config={"method_name": "css", "batch_size": 32}
        )

        # perform_interventions returns results dict
        results = experiment.perform_interventions(
            filtered_datasets,
            target_variables_list=[[variable]],
            save_dir=None
        )

        # 7. Parse Results
        # MIB results structure is complex. We need to extract IIA (Interchange Intervention Accuracy).
        # results["dataset"][dataset_name]["model_unit"][unit_name][variable_name]["average_score"]
        
        scores = []
        per_sample = []
        
        for ds_name, ds_data in results["dataset"].items():
            for unit_name, unit_data in ds_data["model_unit"].items():
                if variable in unit_data:
                     score = unit_data[variable].get("average_score", 0.0)
                     scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkResult(
            task_name=task_name,
            metrics={"iia": avg_score},
            metadata={"num_datasets": len(cf_datasets), "variable": variable}
        )
