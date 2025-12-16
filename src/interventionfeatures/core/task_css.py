"""Task-targeted CSS direction finder for benchmark tasks.

This module implements task-targeted CSS direction finding that optimizes
directly for Interchange Intervention Accuracy (IIA) rather than generic
sensitivity. Instead of finding many features that maximize KL divergence,
it finds the specific direction(s) that maximize task performance.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from .task_data_handler import TaskDataHandler

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@contextmanager
def cuda_memory_manager():
    """Context manager for aggressive memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def get_memory_stats():
    """Get current GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0.0, 0.0


def mics_intervention(
    base_act: torch.Tensor,
    source_act: torch.Tensor,
    s_direction: torch.Tensor,
    alpha: float = 0.875,
) -> torch.Tensor:
    """
    Apply MICS (Minimal Intervention Change Scheme) intervention.

    Swaps the component of the activation along the s direction from
    source to base, while preserving the orthogonal component.

    Args:
        base_act: Base activations (B, d_model)
        source_act: Source activations (B, d_model)
        s_direction: The direction to swap along (d_model,)
        alpha: Scaling factor (default 0.875 = 7/8)

    Returns:
        Intervened activations (B, d_model)
    """
    # Normalize direction
    s_unit = F.normalize(s_direction, p=2, dim=-1)  # (d_model,)

    # Project base onto orthogonal complement of s
    base_parallel = (base_act @ s_unit).unsqueeze(-1) * s_unit  # (B, d_model)
    base_orthogonal = base_act - base_parallel  # (B, d_model)

    # Get source's projection onto s
    source_proj = (source_act @ s_unit).unsqueeze(-1) * s_unit  # (B, d_model)

    # Combine: base orthogonal + scaled source parallel
    intervened = base_orthogonal + alpha * source_proj

    # Normalize to match base norm
    base_norm = torch.norm(base_act, dim=-1, keepdim=True)
    intervened = F.normalize(intervened, dim=-1) * base_norm

    return intervened


class TaskTargetedCSSFinder:
    """
    CSS direction finder optimized for specific benchmark tasks.

    Instead of finding directions that maximize generic sensitivity (KL divergence),
    this finder optimizes directions to maximize task-specific Interchange
    Intervention Accuracy (IIA).

    Key insight:
    - Generic CSS optimizes: "How much does intervention change outputs?"
    - Task-targeted CSS optimizes: "Does intervention cause correct task behavior?"
    """

    def __init__(
        self,
        model: PreTrainedModel,
        task_name: str,
        layer: int,
        device: str = "cuda",
        max_seq_len: int = 128,
        hf_cache_dir: str | None = None,
    ):
        """
        Initialize task-targeted CSS finder.

        Args:
            model: HookedTransformer or compatible model
            task_name: Name of task to optimize for (e.g., "ioi", "ravel_country")
            layer: Layer index for interventions
            device: Device to run on
            max_seq_len: Maximum sequence length
            hf_cache_dir: HuggingFace cache directory
        """
        self.model = model
        self.task_name = task_name
        self.layer = layer
        self.device = device
        self.max_seq_len = max_seq_len

        # Get model dimensions
        if hasattr(model, "cfg"):
            self.d_model = model.cfg.d_model
        elif hasattr(model, "config"):
            self.d_model = model.config.hidden_size
        else:
            raise ValueError("Cannot determine d_model from model")

        # Create task data handler
        self.task_handler = TaskDataHandler(
            task_name=task_name,
            tokenizer=self._get_tokenizer(),
            device=device,
            max_seq_len=max_seq_len,
            hf_cache_dir=hf_cache_dir,
        )

        # Hook name for intervention
        self.hook_name = f"blocks.{layer}.hook_resid_post"

        print("TaskTargetedCSSFinder initialized:")
        print(f"  Task: {task_name}")
        print(f"  Layer: {layer}")
        print(f"  d_model: {self.d_model}")
        print(f"  Device: {device}")
        print(f"  Examples available: {len(self.task_handler)}")

    def _get_tokenizer(self):
        """Get tokenizer from model."""
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer
        elif hasattr(self.model, "get_tokenizer"):
            return self.model.get_tokenizer()
        else:
            raise ValueError("Cannot get tokenizer from model")

    def compute_iia_objective(
        self,
        base_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        s_direction: torch.Tensor,
        intervention_positions: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IIA-based objective for task-targeted optimization.

        Instead of measuring how much outputs change (KL divergence),
        we measure whether outputs change to the correct target.

        Args:
            base_tokens: Base input tokens (B, seq_len)
            source_tokens: Source input tokens (B, seq_len)
            s_direction: Current direction to optimize (d_model,)
            intervention_positions: Positions to intervene (B,)
            target_token_ids: Expected output tokens after intervention (B,)

        Returns:
            Scalar objective value (higher = better IIA)
        """
        batch_size = base_tokens.shape[0]

        # Get activations at intervention positions
        with torch.no_grad():
            _, base_cache = self.model.run_with_cache(
                base_tokens, names_filter=[self.hook_name]
            )
            _, source_cache = self.model.run_with_cache(
                source_tokens, names_filter=[self.hook_name]
            )

        # Extract activations at intervention positions
        base_acts = []
        source_acts = []
        for i in range(batch_size):
            pos = intervention_positions[i].item()
            base_acts.append(base_cache[self.hook_name][i, pos, :])
            source_acts.append(source_cache[self.hook_name][i, pos, :])

        base_act = torch.stack(base_acts, dim=0)  # (B, d_model)
        source_act = torch.stack(source_acts, dim=0)  # (B, d_model)

        # Apply MICS intervention
        intervened_act = mics_intervention(base_act, source_act, s_direction)

        # Create intervention hook
        def intervention_hook(activation, hook):
            for i in range(batch_size):
                pos = intervention_positions[i].item()
                activation[i, pos, :] = intervened_act[i]
            return activation

        # Run model with intervention and get output logits
        with self.model.hooks(fwd_hooks=[(self.hook_name, intervention_hook)]):
            output_logits = self.model(base_tokens)

        # Get logits at final position
        final_logits = output_logits[:, -1, :]  # (B, vocab_size)

        # Compute probability of target tokens
        probs = F.softmax(final_logits, dim=-1)  # (B, vocab_size)

        # Gather probability of target tokens
        target_probs = probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)  # (B,)

        # Also compute log probability for gradient stability
        log_probs = F.log_softmax(final_logits, dim=-1)
        target_log_probs = log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)

        # Compute accuracy (IIA) - does argmax match target?
        predictions = final_logits.argmax(dim=-1)
        accuracy = (predictions == target_token_ids).float().mean()

        # Return log prob for optimization, (accuracy, prob) for display
        return target_log_probs.mean(), (accuracy, target_probs.mean())

    def find_task_direction(
        self,
        num_iterations: int = 512,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        target_norm: float = 10.0,
        eval_freq: int = 50,
    ) -> dict:
        """
        Find the optimal direction for this specific task.

        Uses Projected Gradient Ascent (PGA) to maximize IIA, optimizing
        the direction s that causes interchange interventions to produce
        correct task behavior.

        Args:
            num_iterations: Number of PGA iterations
            batch_size: Batch size for optimization
            learning_rate: Learning rate for optimizer
            target_norm: Target norm for direction vector
            eval_freq: Frequency of evaluation logging

        Returns:
            Dictionary with:
                - s: Found direction tensor (d_model,)
                - iia: Achieved IIA score
                - task: Task name
                - training_history: List of (iteration, iia) tuples
        """
        print(f"\n{'='*60}")
        print(f"Finding task-targeted direction for: {self.task_name}")
        print(f"{'='*60}")

        # Initialize direction randomly on sphere as nn.Parameter
        s_init = torch.randn(self.d_model, device=self.device)
        s_init = F.normalize(s_init, dim=-1) * target_norm
        s = torch.nn.Parameter(s_init)

        # Use standard AdamW with manual projection
        optimizer = AdamW([s], lr=learning_rate)

        training_history = []
        best_iia = 0.0
        best_s = s.detach().clone()

        pbar = tqdm(
            range(num_iterations),
            desc=f"Task-CSS ({self.task_name})",
            file=sys.stderr,
        )

        for iteration in pbar:
            with cuda_memory_manager():
                # Get batch of contrastive examples
                batch = self.task_handler.get_contrastive_batch(batch_size)

                # Compute IIA objective
                log_prob_obj, (accuracy, prob) = self.compute_iia_objective(
                    base_tokens=batch["base_tokens"],
                    source_tokens=batch["source_tokens"],
                    s_direction=s,
                    intervention_positions=batch["intervention_positions"],
                    target_token_ids=batch["source_answer_tokens"],
                )

                # Maximize log probability via gradient ascent
                loss = -log_prob_obj  # Negative because we maximize

                # Compute gradient and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Project back to sphere (maintain norm constraint)
                with torch.no_grad():
                    s.data = F.normalize(s.data, dim=-1) * target_norm

                # Track best (using accuracy as IIA)
                current_iia = accuracy.item()
                current_prob = prob.item()
                if current_iia > best_iia:
                    best_iia = current_iia
                    best_s = s.detach().clone()

                # Log progress
                if iteration % eval_freq == 0 or iteration == num_iterations - 1:
                    training_history.append((iteration, current_iia))

                    allocated, _ = get_memory_stats()
                    pbar.set_postfix({
                        "IIA": f"{current_iia:.2f}",
                        "prob": f"{current_prob:.3f}",
                        "best": f"{best_iia:.2f}",
                        "GPU": f"{allocated:.1f}G",
                    })

        # Final evaluation
        final_iia = self._evaluate_full_iia(best_s, min(100, len(self.task_handler)))

        print(f"\nTraining completed for task: {self.task_name}")
        print(f"  Best training IIA: {best_iia:.4f}")
        print(f"  Final evaluation IIA: {final_iia:.4f}")
        print(f"  Direction norm: {torch.norm(best_s).item():.2f}")

        return {
            "s": best_s.detach().cpu(),
            "iia": final_iia,
            "task": self.task_name,
            "layer": self.layer,
            "training_history": training_history,
            "d_model": self.d_model,
        }

    def _evaluate_full_iia(
        self,
        s_direction: torch.Tensor,
        num_samples: int = 100,
    ) -> float:
        """
        Evaluate IIA on a larger sample for final score.

        Args:
            s_direction: Direction to evaluate
            num_samples: Number of samples to evaluate

        Returns:
            IIA score (0.0 to 1.0)
        """
        total_correct = 0
        total_samples = 0
        # Use larger batch for evaluation (no gradients needed)
        batch_size = 64

        with torch.no_grad():
            for _ in range(num_samples // batch_size + 1):
                if total_samples >= num_samples:
                    break

                batch = self.task_handler.get_contrastive_batch(
                    min(batch_size, num_samples - total_samples)
                )

                # Get activations
                _, base_cache = self.model.run_with_cache(
                    batch["base_tokens"], names_filter=[self.hook_name]
                )
                _, source_cache = self.model.run_with_cache(
                    batch["source_tokens"], names_filter=[self.hook_name]
                )

                bs = batch["batch_size"]
                base_acts = []
                source_acts = []
                for i in range(bs):
                    pos = batch["intervention_positions"][i].item()
                    base_acts.append(base_cache[self.hook_name][i, pos, :])
                    source_acts.append(source_cache[self.hook_name][i, pos, :])

                base_act = torch.stack(base_acts, dim=0)
                source_act = torch.stack(source_acts, dim=0)

                # Apply intervention
                intervened_act = mics_intervention(base_act, source_act, s_direction.to(self.device))

                # Create hook - capture variables via default arguments
                intervention_positions = batch["intervention_positions"]

                def intervention_hook(
                    activation,
                    hook,
                    _bs=bs,
                    _positions=intervention_positions,
                    _intervened=intervened_act,
                ):
                    for i in range(_bs):
                        pos = _positions[i].item()
                        activation[i, pos, :] = _intervened[i]
                    return activation

                # Run with intervention
                with self.model.hooks(fwd_hooks=[(self.hook_name, intervention_hook)]):
                    output_logits = self.model(batch["base_tokens"])

                # Check predictions
                predictions = output_logits[:, -1, :].argmax(dim=-1)
                targets = batch["source_answer_tokens"]

                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += bs

        return total_correct / total_samples if total_samples > 0 else 0.0

    def find_multiple_directions(
        self,
        num_directions: int = 3,
        num_iterations: int = 512,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        target_norm: float = 10.0,
        orthogonality_weight: float = 0.1,
    ) -> list[dict]:
        """
        Find multiple orthogonal directions for the task.

        Uses deflation: after finding each direction, adds orthogonality
        penalty to encourage finding new directions.

        Args:
            num_directions: Number of directions to find
            num_iterations: Iterations per direction
            batch_size: Batch size
            learning_rate: Learning rate
            target_norm: Target norm
            orthogonality_weight: Weight for orthogonality penalty

        Returns:
            List of direction dictionaries
        """
        results = []
        prior_directions = []

        for i in range(num_directions):
            print(f"\n{'='*60}")
            print(f"Finding direction {i+1}/{num_directions} for task: {self.task_name}")
            print(f"{'='*60}")

            result = self._find_direction_with_prior(
                prior_directions=prior_directions,
                num_iterations=num_iterations,
                batch_size=batch_size,
                learning_rate=learning_rate,
                target_norm=target_norm,
                orthogonality_weight=orthogonality_weight,
            )

            results.append(result)
            prior_directions.append(result["s"].to(self.device))

        return results

    def _find_direction_with_prior(
        self,
        prior_directions: list[torch.Tensor],
        num_iterations: int,
        batch_size: int,
        learning_rate: float,
        target_norm: float,
        orthogonality_weight: float,
    ) -> dict:
        """Find direction while penalizing overlap with prior directions."""
        # Initialize direction
        s = torch.randn(self.d_model, device=self.device)
        s = F.normalize(s, dim=-1) * target_norm
        s.requires_grad_(True)

        optimizer = AdamW([s], lr=learning_rate)

        best_iia = 0.0
        best_s = s.detach().clone()

        pbar = tqdm(range(num_iterations), desc="Task-CSS", file=sys.stderr)

        for iteration in pbar:
            with cuda_memory_manager():
                batch = self.task_handler.get_contrastive_batch(batch_size)

                log_prob_obj, prob_obj = self.compute_iia_objective(
                    base_tokens=batch["base_tokens"],
                    source_tokens=batch["source_tokens"],
                    s_direction=s,
                    intervention_positions=batch["intervention_positions"],
                    target_token_ids=batch["source_answer_tokens"],
                )

                # Add orthogonality penalty
                ortho_penalty = 0.0
                if prior_directions:
                    s_unit = F.normalize(s, dim=-1)
                    for prior in prior_directions:
                        prior_unit = F.normalize(prior, dim=-1)
                        overlap = (s_unit @ prior_unit) ** 2
                        ortho_penalty += overlap

                # Total loss: maximize IIA, minimize overlap
                loss = -log_prob_obj + orthogonality_weight * ortho_penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Project back to sphere
                with torch.no_grad():
                    s.data = F.normalize(s.data, dim=-1) * target_norm

                current_iia = prob_obj.item()
                if current_iia > best_iia:
                    best_iia = current_iia
                    best_s = s.detach().clone()

                if iteration % 50 == 0:
                    pbar.set_postfix({
                        "IIA": f"{current_iia:.4f}",
                        "ortho": f"{ortho_penalty:.4f}",
                    })

        final_iia = self._evaluate_full_iia(best_s, 100)

        return {
            "s": best_s.detach().cpu(),
            "iia": final_iia,
            "task": self.task_name,
            "layer": self.layer,
            "d_model": self.d_model,
        }
