"""
SAE Benchmark Runner

Evaluates pre-trained SAEs on MIB/RAVEL benchmarks for comparison with CSS directions.
Uses SAE-lens library for loading SAEs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from sae_lens import SAE

from .base import BaseBenchmarkRunner, BenchmarkResult


class SAEFeaturizer(nn.Module):
    """
    Wraps an SAE for use as a featurizer in interchange interventions.

    The SAE provides:
    - encode(x) -> latent activations
    - decode(latents) -> reconstructed activations

    For interchange intervention, we:
    1. Encode base and source activations
    2. Swap specified latent dimensions
    3. Decode back to activation space
    """

    def __init__(self, sae: SAE, latent_indices: list[int] | None = None):
        super().__init__()
        self.sae = sae
        self.latent_indices = latent_indices  # Which SAE latents to intervene on

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to SAE latent space."""
        # SAE expects (batch, d_model)
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        # Get SAE latents
        latents = self.sae.encode(x_flat)

        return latents.view(*original_shape[:-1], -1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode SAE latents back to activation space."""
        original_shape = latents.shape
        latents_flat = latents.view(-1, latents.shape[-1])

        reconstructed = self.sae.decode(latents_flat)

        return reconstructed.view(*original_shape[:-1], -1)

    def forward(
        self,
        base_act: torch.Tensor,
        source_act: torch.Tensor,
        latent_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Perform interchange intervention using SAE latents.

        Args:
            base_act: Base activations (batch, d_model)
            source_act: Source activations (batch, d_model)
            latent_indices: Which latents to swap (None = use self.latent_indices or all)

        Returns:
            Intervened activations (batch, d_model)
        """
        indices = latent_indices or self.latent_indices

        # Encode both
        base_latents = self.encode(base_act)
        source_latents = self.encode(source_act)

        # Swap specified latent dimensions
        intervened_latents = base_latents.clone()
        if indices is not None:
            for idx in indices:
                intervened_latents[..., idx] = source_latents[..., idx]
        else:
            # Swap all latents (full replacement)
            intervened_latents = source_latents

        # Decode back
        return self.decode(intervened_latents)


class SAEBenchmarkRunner(BaseBenchmarkRunner):
    """
    Benchmark runner for SAE-based features.

    Loads pre-trained SAEs from SAE-lens and evaluates them on MIB/RAVEL tasks.
    """

    # Available SAE configurations for pythia-70m
    SAE_CONFIGS = {
        "pythia-70m-res": {
            "release": "pythia-70m-deduped-res-sm",
            "sae_id_template": "blocks.{layer}.hook_resid_post",
        },
        "pythia-70m-mlp": {
            "release": "pythia-70m-deduped-mlp-sm",
            "sae_id_template": "blocks.{layer}.hook_mlp_out",
        },
    }

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-70m-deduped",
        layer: int = 2,
        sae_type: str = "pythia-70m-res",
        latent_indices: list[int] | None = None,
        device: str = "cuda",
    ):
        """
        Initialize SAE benchmark runner.

        Args:
            model_name: HuggingFace model name
            layer: Layer to evaluate
            sae_type: SAE configuration key (from SAE_CONFIGS)
            latent_indices: Specific SAE latents to use for interventions
            device: Device to run on
        """
        # We don't have CSS directions, pass empty list
        super().__init__(
            css_directions=[],
            model_name=model_name,
            layer=layer,
            device=device,
        )

        self.sae_type = sae_type
        self.latent_indices = latent_indices
        self._sae: SAE | None = None
        self._sae_featurizer: SAEFeaturizer | None = None

    def _load_sae(self) -> SAE:
        """Load SAE from SAE-lens."""
        if self._sae is None:
            config = self.SAE_CONFIGS.get(self.sae_type)
            if config is None:
                raise ValueError(f"Unknown SAE type: {self.sae_type}")

            sae_id = config["sae_id_template"].format(layer=self.layer)
            print(f"Loading SAE: {config['release']} / {sae_id}")

            self._sae = SAE.from_pretrained(
                release=config["release"],
                sae_id=sae_id,
                device=self.device,
            )[0]  # Returns (sae, cfg, sparsity)

            print(f"SAE loaded: {self._sae.cfg.d_sae} latents")

        return self._sae

    def get_featurizer(self, latent_indices: list[int] | None = None) -> SAEFeaturizer:
        """Get SAE featurizer for interventions."""
        sae = self._load_sae()
        indices = latent_indices or self.latent_indices
        return SAEFeaturizer(sae, latent_indices=indices)

    def load_data(self, task_name: str) -> Any:
        """Load task data - delegates to task-specific loaders."""
        return None

    def get_available_tasks(self) -> list[str]:
        """Return list of available tasks."""
        return ["ioi", "ravel_cities"]

    def run_evaluation(
        self,
        task_name: str,
        latent_indices: list[int] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run SAE evaluation on a task.

        For now, this is a simplified evaluation that tests whether SAE latents
        can perform interchange interventions.
        """
        from tqdm import tqdm

        # Load model
        if self._model is None:
            self._load_model()
        model = self._model

        # Get SAE featurizer
        featurizer = self.get_featurizer(latent_indices)
        featurizer = featurizer.to(self.device)

        hook_name = f"blocks.{self.layer}.hook_resid_post"

        if task_name == "ioi":
            return self._eval_ioi(model, featurizer, hook_name, num_samples)
        elif task_name.startswith("ravel"):
            return self._eval_ravel(model, featurizer, hook_name, task_name, num_samples)
        else:
            return BenchmarkResult(
                task_name=task_name,
                metrics={"error": "Unknown task"},
                metadata={}
            )

    def _eval_ioi(
        self,
        model,
        featurizer: SAEFeaturizer,
        hook_name: str,
        num_samples: int | None = None,
    ) -> BenchmarkResult:
        """Evaluate on IOI task using mib-bench/ioi dataset."""
        from datasets import load_dataset
        from tqdm import tqdm

        # Load IOI dataset
        ds = load_dataset("mib-bench/ioi", split="test")
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))

        correct = 0
        total = 0
        filtered_out = 0

        for example in tqdm(ds, desc="IOI (SAE)"):
            # Base input and expected answer
            base_input = example["prompt"]
            base_choices = example["choices"]
            base_answer_idx = example["answerKey"]
            base_expected = base_choices[base_answer_idx]

            # Source input (use s1_io_flip as counterfactual)
            cf = example["s1_io_flip_counterfactual"]
            source_input = cf["prompt"]
            source_answer_idx = cf["answerKey"]
            source_expected = cf["choices"][source_answer_idx]

            # Tokenize
            base_tokens = model.tokenizer(base_input, return_tensors="pt")
            base_ids = base_tokens["input_ids"].to(self.device)
            source_tokens = model.tokenizer(source_input, return_tensors="pt")
            source_ids = source_tokens["input_ids"].to(self.device)

            # Check if model gets base correct (filtering)
            with torch.no_grad():
                base_out = model.generate(
                    base_ids,
                    max_new_tokens=3,
                    do_sample=False,
                )
            base_pred = model.tokenizer.decode(
                base_out[0, base_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            if base_expected not in base_pred:
                filtered_out += 1
                continue

            # Get activations at last token
            with torch.no_grad():
                _, base_cache = model.run_with_cache(
                    base_ids,
                    names_filter=[hook_name]
                )
                _, source_cache = model.run_with_cache(
                    source_ids,
                    names_filter=[hook_name]
                )

            base_act = base_cache[hook_name][:, -1, :]
            source_act = source_cache[hook_name][:, -1, :]

            # Apply SAE intervention (swap ALL latents)
            intervened_act = featurizer(base_act, source_act)

            # Run with intervention
            prompt_len = base_ids.shape[1]

            def make_hook(i_act, p_len):
                def hook(activation, hook=None):
                    if activation.shape[1] == p_len:
                        activation[:, -1, :] = i_act
                    return activation
                return hook

            with model.hooks(fwd_hooks=[(hook_name, make_hook(intervened_act, prompt_len))]):
                output_ids = model.generate(
                    base_ids,
                    max_new_tokens=3,
                    do_sample=False,
                )

            predicted = model.tokenizer.decode(
                output_ids[0, base_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # After intervention, should predict source's answer
            if source_expected in predicted:
                correct += 1
            total += 1

        iia = correct / total if total > 0 else 0.0

        return BenchmarkResult(
            task_name="ioi",
            metrics={"iia": iia},
            metadata={
                "sae_type": self.sae_type,
                "num_latents": self._sae.cfg.d_sae if self._sae else 0,
                "examples_evaluated": total,
                "examples_filtered": filtered_out,
                "intervention": "full_sae" if self.latent_indices is None else f"{len(self.latent_indices)}_latents",
            }
        )

    def _eval_ravel(
        self,
        model,
        featurizer: SAEFeaturizer,
        hook_name: str,
        task_name: str,
        num_samples: int | None = None,
    ) -> BenchmarkResult:
        """Evaluate on RAVEL task - stub for now."""
        return BenchmarkResult(
            task_name=task_name,
            metrics={"cause": 0.0, "isolation": 0.0, "disentangle": 0.0},
            metadata={"note": "RAVEL evaluation not yet implemented for SAE"}
        )

    def run_all_evaluations(
        self,
        latent_indices: list[int] | None = None,
        tasks: list[str] | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> dict[str, BenchmarkResult]:
        """Run all available evaluations."""
        if tasks is None:
            tasks = self.get_available_tasks()

        results = {}
        for task in tasks:
            print(f"\nEvaluating {task} with SAE...")
            try:
                results[task] = self.run_evaluation(
                    task,
                    latent_indices=latent_indices,
                    num_samples=num_samples,
                    **kwargs
                )
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                results[task] = BenchmarkResult(
                    task_name=task,
                    metrics={"error": str(e)},
                    metadata={}
                )

        return results
