import math

import torch
import torch.nn as nn

EVAL_SAMPLE = 128
import sys
from contextlib import contextmanager

import torch.nn.functional as F
from tqdm.auto import tqdm

from ..utils.math_utils import robust_kl_divergence_batched
from .data_handler import TransformerDataHandler
from .model import IntervenableTransformerSegment
from .scheduler import (
    CosineAnnealingScheduler,
    ExponentialDecayScheduler,
    LearningRateScheduler,
    LinearDecayScheduler,
    SphericalAdamW,
    StepDecayScheduler,
)


# Method 2: Using torch.multinomial (often fastest)
def sample_multinomial(N, K, token_offset: int, sorted=True):
    """
    We want to have some order, so we have samples which are sorted.
    """
    # Create uniform probabilities
    token_offset = 0 if token_offset <= 0 else token_offset
    probs = torch.ones(N - token_offset) / (N - token_offset)
    r = torch.multinomial(probs, K, replacement=False).type(torch.int64)
    if sorted:
        r, _ = r.sort(dim=-1)  # Sort the last dimensionin ascending order
    return r


def select_positions(x_batch: torch.Tensor, pos_indices: torch.Tensor):
    x_sel = torch.zeros_like(
        x_batch[:, torch.arange(pos_indices.shape[-1])], device=x_batch.device
    )
    for b_idx in range(x_batch.shape[0]):
        x_sel[b_idx] = x_batch[b_idx, pos_indices[b_idx]]
    return x_sel


def calc_overlap(s: torch.Tensor, s_prior: torch.Tensor, do_relu=True):
    c = F.cosine_similarity(s.unsqueeze(0), s_prior, dim=-1) ** 2
    if not do_relu:
        return c
    return F.relu(c)


# Memory management utilities (keeping your existing ones)
@contextmanager
def cuda_memory_manager():
    """Context manager for aggressive memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def get_memory_stats():
    """Get current GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0.0, 0.0


def intervention_objective(
    fx: dict[str, torch.Tensor],
    fy: dict[str, torch.Tensor],
    x,
    s_current: torch.Tensor,  # Shape (S, K, d_model)
    s_prior: torch.Tensor,  # Shape (K, d_model)
    overlap_penalty: float = 100.0,  # 0.5,
    target_norm: float = 1.0,
    norm_penalty=1.0,
) -> torch.Tensor:
    """
    We expect fx and fy to be dictionaries with keys corresponding to different layers or components.
    Each tensor should have shape (batch_size, K, d_model) where K is the number of coordinates

    s_current should have shape (K, d_model)
    """
    """Compute L2 objective between two tensors"""

    all_fx: list[torch.Tensor] = [
        fx[k] for k in fx.keys()
    ]  # Unsqueeze for dimension matching to S
    all_fy: list[torch.Tensor] = [fy[k] for k in fy.keys()]

    BS = all_fx[0].shape[0]

    if len(all_fx) != len(all_fy) and not all(
        [x.shape == all_fy[i].shape for i, x in all_fx.enumerate()]
    ):
        raise ValueError("fx and fy must have the same shape")

    rets = torch.zeros((BS, len(all_fx)), device=all_fx[0].device)
    for i, (k, fx, fy) in enumerate(zip(fx.keys(), all_fx, all_fy)):
        if k == "final_logits":
            r = robust_kl_divergence_batched(fx, fy)
            rets[:, i] = r
        else:
            r = ((1 - F.cosine_similarity(fx, fy, dim=-1)) / 2)
            r = r.squeeze(-1)
            rets[:, i] = r

    r = torch.sum(
        F.softmax(rets, dim=-1) * rets, dim=-1
    )  # Take the softmax to get the "most change"

    corrections = torch.zeros(BS, device=all_fx[0].device)
    if (
        overlap_penalty > 0.0 and s_prior.shape[0] > 0
    ):  # Allow for a "quick start via r.mean() > 0.1"
        overlap = calc_overlap(s_current, s_prior, do_relu=True)
        overlap = torch.sum(
            F.softmax(overlap, dim=0) * overlap, dim=0
        )  # Take the (soft) max over the maximum positive overlap, get shape [BS, K]
        overlap = torch.sum(
            F.softmax(overlap, dim=-1) * overlap
        )  # Take the (soft) max over the K possible items
        assert torch.where(overlap < 0, 1.0, 0.0).sum() == 0.0
        # Scale overlap by r so that we have a relative result
        corrections -= overlap_penalty * overlap * torch.abs(r)

    if norm_penalty > 0.0:
        current_norm = torch.linalg.norm(s_current, dim=-1).mean()
        # Calculate the squared difference from the target
        norm_error = (current_norm - target_norm) ** 2
        # The penalty is this squared error, scaled by your hyperparameter
        n = norm_penalty * norm_error
        # Subtract it from the objective
        corrections -= n

    return (
        r + corrections
    )


def add_in_offset_vec(
    x_batch_detached: torch.Tensor,
    s_current: torch.Tensor,
    alpha: torch.Tensor | float,
    use_MICS: bool = True,
):
    """
    Shape of s_current is (K, d_model)
    Shape of x_batch_detached is (B, K, d_model) where B is the batch size and K is the number of coordinates.
    Assumes that ||s_current[i]|| = 1.0 for all i in [0, K-1]

    Returns a tensor of shape (B, K, d_model) which is the modified x_batch_detached
    """
    # Set the inner product

    if use_MICS:
        alpha = 7 * alpha / 8
        s_unit = F.normalize(s_current, p=2, dim=-1).unsqueeze(0)

        # Create a boolean mask where the condition |cos_sim| <= eps is met
        # mask = torch.abs(cos_sim) <= 0.3

        # Decompose x into parallel and orthogonal components with respect to s_unit
        x_parallel = torch.sum(x_batch_detached * s_unit,
                               dim=-1, keepdim=True) * s_unit
        x_orthogonal = x_batch_detached - x_parallel

        # Calculate the new parallel component using the derived formula
        # An epsilon is added to the denominator to avoid division by zero if alpha is +/-1
        c_times_s_norm = torch.linalg.norm(x_orthogonal, dim=-1, keepdim=True) * alpha
        new_parallel_component = (c_times_s_norm / torch.sqrt(1.0 - alpha**2 + 1e-8)) * s_unit

        # Reconstruct the new vector
        x_prime = new_parallel_component + x_orthogonal
        x_final = F.normalize(x_prime, dim=-1)
        return x_final

    else:
        return x_batch_detached + s_current.unsqueeze(0)

def adaptive_batch_size(base_batch_size: int, max_memory_gb: float = 0.8) -> int:
    """Dynamically adjust batch size based on available memory"""
    if not torch.cuda.is_available():
        return base_batch_size

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated, reserved = get_memory_stats()
    available = total_memory - allocated

    if available < total_memory * max_memory_gb:
        return max(1, base_batch_size // 2)
    return base_batch_size


# Enhanced CSS Direction Finder
class CSSDirectionFinder:
    @classmethod
    def from_config(
        cls,
        config,
        data_handler: TransformerDataHandler,
        model_segment: IntervenableTransformerSegment,
    ):
        """Create CSSDirectionFinder from a config object."""
        return cls(
            data_handler=data_handler,
            model_segment=model_segment,
            pga_batch_size=config.training.pga_batch_size,
            eval_batch_size=config.training.eval_batch_size,
            n_norm_discretization_steps=config.model.n_norm_discretization_steps,
            num_x_samples_for_final_eval=5 * EVAL_SAMPLE,
            dict_size=config.training.dict_size,
            pga_iterations=config.training.pga_its,
            pga_learning_rate=config.training.learning_rate,
            sample_temp=config.training.sample_temp,
            target_norm=config.model.target_norm,
            kwise_coords=config.training.k,
            norm_lower_bound=config.training.norm_lower_bound,
            norm_upper_bound=config.training.norm_upper_bound,
            optimizer_type="adamw",
            scheduler_params={
                "betas": (config.training.beta1, config.training.beta2),
                "eps": config.training.eps,
                "weight_decay": config.training.weight_decay,
            },
            early_stopping_enabled=config.training.early_stopping.enabled,
            early_stopping_patience=config.training.early_stopping.patience,
            early_stopping_min_delta=config.training.early_stopping.min_delta,
            early_stopping_eval_freq=config.training.early_stopping.eval_freq,
            use_MICS=config.database.use_MICS,
        )

    def __init__(
        self,
        data_handler: TransformerDataHandler,
        model_segment: IntervenableTransformerSegment,
        max_overlap: float = 0.1,
        pga_batch_size: int = 32,
        eval_batch_size: int = 33,
        intermediate_eval: bool = True,
        num_x_samples_for_final_eval: int = 200,
        dict_size: int = 5,
        pga_iterations: int = 50,
        pga_learning_rate: float = 0.05,
        max_memory_usage: float = 0.85,
        use_gradient_checkpointing: bool = True,
        sample_temp: float = 0.5,
        target_norm: float = 10.0,
        kwise_coords: int = 2,
        norm_lower_bound: float = 0.5,
        norm_upper_bound: float = 90.0,
        optimizer_type: str = "adamw",
        scheduler_type: str = "exponential",
        scheduler_params: dict | None = None,
        early_stopping_enabled: bool = False,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-6,
        early_stopping_eval_freq: int = 10,
        n_norm_discretization_steps: int = 1,
        use_MICS: bool = True,
    ):
        self.n_norm_discretization_steps = n_norm_discretization_steps
        self.data_handler = data_handler
        self.max_overlap = max_overlap
        self.model_segment = model_segment
        self.max_memory_usage = max_memory_usage
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.kwise_coordinates = kwise_coords
        self.target_token_offset = int(model_segment.target_token_offset)
        self.intermediate_eval = intermediate_eval
        self.sample_temp = sample_temp
        self.target_norm = target_norm
        self.use_MICS = use_MICS
        self.norm_lower_bound = norm_lower_bound
        self.norm_upper_bound = norm_upper_bound
        self.pga_batch_size = pga_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_x_samples_for_final_eval = num_x_samples_for_final_eval
        self.dict_size = dict_size
        self.pga_iterations = pga_iterations
        self.pga_learning_rate = pga_learning_rate
        self.d_model = data_handler.d_model
        self.device = self._determine_device()
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_eval_freq = early_stopping_eval_freq
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}

        self._setup_model_for_efficiency()
        print("CSSFinder initialized:")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: {self.optimizer_type}")
        print(f"  Scheduler: {self.scheduler_type}")
        print(f"  Gradient Checkpointing: {self.use_gradient_checkpointing}")
        print(f"  Max Memory Usage: {self.max_memory_usage:.2%}")
        print(f"  Early Stopping: {self.early_stopping_enabled}")
        if self.early_stopping_enabled:
            print(f"    Patience: {self.early_stopping_patience}")
            print(f"    Min Delta: {self.early_stopping_min_delta}")
            print(f"    Eval Frequency: {self.early_stopping_eval_freq}")

    def _determine_device(self):
        """Determine the best device to use"""
        if (
            hasattr(self.data_handler, "device")
            and self.data_handler.device is not None
        ):
            return self.data_handler.device
        elif (
            hasattr(self.model_segment, "device")
            and self.model_segment.device is not None
        ):
            return self.model_segment.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"CSSFinder: device not explicitly found, defaulting to {device}")
            return device

    def _setup_model_for_efficiency(self):
        """Setup model for memory efficiency"""
        if hasattr(self.model_segment, "base_model") and isinstance(
            self.model_segment.base_model, nn.Module
        ):
            self.model_segment.base_model.eval()

            # Disable gradients for base model parameters
            for param in self.model_segment.base_model.parameters():
                param.requires_grad = False

            # Enable gradient checkpointing if requested
            if self.use_gradient_checkpointing and hasattr(
                self.model_segment.base_model, "gradient_checkpointing_enable"
            ):
                try:
                    self.model_segment.base_model.gradient_checkpointing_enable()
                    print("Gradient checkpointing enabled for base model")
                except:
                    print("Warning: Could not enable gradient checkpointing")

            print("Base model optimized for memory efficiency")

    def _create_scheduler(self) -> LearningRateScheduler:
        """Create learning rate scheduler"""
        if self.scheduler_type == "exponential":
            decay_rate = self.scheduler_params.get("decay_rate", 0.995)
            return ExponentialDecayScheduler(self.pga_learning_rate, decay_rate)
        elif self.scheduler_type == "cosine":
            min_lr = self.scheduler_params.get("min_lr", 1e-6)
            return CosineAnnealingScheduler(
                self.pga_learning_rate, self.pga_iterations, min_lr
            )
        elif self.scheduler_type == "step":
            step_size = self.scheduler_params.get("step_size", self.pga_iterations // 4)
            gamma = self.scheduler_params.get("gamma", 0.5)
            return StepDecayScheduler(self.pga_learning_rate, step_size, gamma)
        elif self.scheduler_type == "linear":
            min_lr = self.scheduler_params.get("min_lr", 1e-6)
            return LinearDecayScheduler(
                self.pga_learning_rate, self.pga_iterations, min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def _create_optimizer(self):
        """Create optimizer"""
        if self.optimizer_type == "adam":
            raise NotImplementedError(
                "Adam optimizer is not supported in this version. Use AdamW or SphericalAdamW instead."
            )
        if self.optimizer_type == "adamw":
            betas = self.scheduler_params.get("betas", (0.9, 0.999))
            eps = self.scheduler_params.get("eps", 1e-8)
            weight_decay = self.scheduler_params.get("weight_decay", 0.01)
            return SphericalAdamW(
                lr=self.pga_learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay)
        elif self.optimizer_type == "sgd_decay":
            raise NotImplementedError(
                "SGD optimizer is not supported in this version. Use SphericalAdam or SphericalAdamW instead."
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _check_early_stopping(
        self, score_history: list[float], patience_counter: int
    ) -> tuple[bool, int]:
        """Check if early stopping criteria are met.

        Args:
            score_history: List of recent evaluation scores
            patience_counter: Current patience counter

        Returns:
            Tuple of (should_stop, updated_patience_counter)
        """
        if not self.early_stopping_enabled or len(score_history) < 2:
            return False, patience_counter

        # Check if current score improved by at least min_delta
        current_score = score_history[-1]
        best_score = max(score_history[:-1])  # Best score so far (excluding current)

        improvement = current_score - best_score

        if improvement >= self.early_stopping_min_delta:
            # Improvement found, reset patience
            return False, 0
        else:
            # No significant improvement
            patience_counter += 1
            if patience_counter >= self.early_stopping_patience:
                return True, patience_counter
            return False, patience_counter

    def _generate_s_samples(self, K: int) -> torch.Tensor:
        """Generate random samples on unit sphere"""
        with cuda_memory_manager():
            s_rand = torch.randn(
                K, self.d_model, device=self.device, dtype=torch.float32
            )
            s_rand_norm = torch.linalg.norm(s_rand, dim=-1, keepdim=True)
            s_on_unit_sphere = torch.div(s_rand, s_rand_norm + 1e-9)
            return s_on_unit_sphere * self.target_norm

    def _generate_batch_samples(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of samples from the data handler.

        Returns:
            x_batch: Tensor of shape (batch_size, seq_len, d_model)
            tokens_batch_padded: Padded tensor of token IDs
            pos_indices: Tensor of shape (batch_size, kwise_coordinates) with random positions for injection
        """
        (
            x_batch_from_handler,
            tokens_batch_list,
        ) = self.data_handler.get_batch(batch_size)

        if x_batch_from_handler.shape[0] == 0:
            raise ValueError(
                "No data returned from data handler. Check your dataset and batch size."
            )

        x_batch = x_batch_from_handler.to(self.device, non_blocking=True)

        # For every item in the batch, sample random positions for injection
        pos_indices = torch.zeros(
            (x_batch.shape[0], self.kwise_coordinates),
            dtype=torch.int64,
            device=self.device,
        )
        for b_idx in range(x_batch.shape[0]):
            pos_indices[b_idx] = sample_multinomial(
                x_batch[b_idx].shape[0],
                self.kwise_coordinates,
                self.target_token_offset,
            )

        tokens_batch_padded = self._pad_and_stack_tokens_optimized(
            tokens_batch_list, self.data_handler.tokenizer, self.device
        )
        return x_batch, tokens_batch_padded, pos_indices

    def _get_alpha(self, batch_size: int):
        alpha = torch.rand(
            (batch_size, 1, 1), device=self.device
        )
        return alpha

    def _pga_iteration_enhanced(
        self,
        s_current: torch.Tensor,
        optimizer,
        scheduler,
        iteration: int,
        s_prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhanced PGA iteration with optimizers and schedulers

        s_current: Shape of (S, self.kwise_coordinates, d_model)

        """
        # Adaptive batch size
        current_batch_size = adaptive_batch_size(
            self.pga_batch_size, self.max_memory_usage
        )

        with cuda_memory_manager():
            # Get batch data
            (
                x_batch,
                tokens_batch_padded,
                pos_indices,
            ) = self._generate_batch_samples(current_batch_size)

            if tokens_batch_padded.numel() == 0:
                raise ValueError(
                    "No tokens returned from data handler. Check your dataset and batch size."
                )

            # Compute gradient with mixed precision and checkpointing
            # Detach x_batch and compute norms
            s_current.requires_grad_(True)
            x_sel = select_positions(x_batch, pos_indices)

            alpha = self._get_alpha(x_sel.shape[0])  # Samlpe a random scaling value

            # y_batch_embed = add_in_offset_vec(x_sel, s_current, alpha)
            y_batch_embed = add_in_offset_vec(x_sel, s_current, alpha, self.use_MICS)

            with torch.no_grad():
                fx_logits = self.model_segment(x_sel, tokens_batch_padded, pos_indices)

            fy_logits = self.model_segment(
                y_batch_embed, tokens_batch_padded, pos_indices
            )

            objective = intervention_objective(
                fx_logits,
                fy_logits,
                x_sel,
                s_current,
                s_prior.detach(),
                target_norm=self.target_norm,
            )

            objective = objective.mean()  # Average over batch

            # Compute gradient
            grad_s = torch.autograd.grad(objective, s_current)[0]

            # Update s_current using optimizer or scheduler
            if optimizer is not None:
                # Use Adam-based optimizer
                new_vals = optimizer.step(s_current, grad_s)
                s_current.detach_()
                s_current = new_vals
            else:
                raise ValueError("Optimizer being none is no longer supported")
                # Use SGD with scheduler
                current_lr = scheduler.step(iteration)
                s_prime = s_current + current_lr * grad_s
                s_prime_norm = torch.linalg.norm(s_prime, dim=-1).unsqueeze(-1)
                s_current.detach_()

                if s_prime_norm > 1e-9:
                    s_current = (s_prime / s_prime_norm) * self.norm_max
                else:
                    raise ValueError(
                        "s_prime norm is too small, cannot normalize. There may be a bug here!"
                    )

            # Cleanup
            del x_batch, tokens_batch_padded, x_sel
            del y_batch_embed, fx_logits, fy_logits, objective, grad_s
            if "fx" in locals():
                del fx, fy

        return s_current.detach()

    def _pad_and_stack_tokens_optimized(self, tokens_list, tokenizer, device_to_use):
        """Memory-optimized version of pad and stack"""
        pad_id = self._get_pad_token_id_from_tokenizer(tokenizer)
        # Process tokens more efficiently
        processed_tokens = []
        max_len = 0

        for t in tokens_list:
            if isinstance(t, torch.Tensor):
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                elif t.ndim == 0:
                    t = torch.tensor([[pad_id]], dtype=torch.long, device=device_to_use)
                processed_tokens.append(t)
                if t.shape[1] > max_len:
                    max_len = t.shape[1]
            else:
                # Handle non-tensor inputs
                processed_tokens.append(
                    torch.tensor([[pad_id]], dtype=torch.long, device=device_to_use)
                )

        if max_len == 0:
            return torch.empty(
                len(processed_tokens), 0, dtype=torch.long, device=device_to_use
            )

        # Pre-allocate output tensor
        batch_size = len(processed_tokens)
        result = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long, device=device_to_use
        )

        # Fill in-place
        for i, t in enumerate(processed_tokens):
            seq_len = min(t.shape[1], max_len)
            result[i, :seq_len] = t[0, :seq_len]

        return result

    def _get_pad_token_id_from_tokenizer(self, tokenizer):
        if tokenizer.pad_token_id is not None:
            return tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            return tokenizer.eos_token_id
        tqdm.write(
            "Warning: pad_token_id and eos_token_id are None. Using 0 as pad_token_id."
        )
        return 0

    def _find_optimal_s_single(self, prior_s: torch.Tensor, pga_its=None) -> dict:
        if pga_its is None:
            pga_its = self.pga_iterations
        print("PRIOR S SHAPE", prior_s.shape, prior_s.device)
        # Run PGA instances
        with cuda_memory_manager():
            s_current = self._generate_s_samples(self.kwise_coordinates)

            # Create optimizer and scheduler for this run
            optimizer = self._create_optimizer()
            scheduler = self._create_scheduler() if optimizer is None else None

            # Reset optimizer state for each run
            if optimizer is not None:
                optimizer.reset()

            # Early stopping state
            score_history = []
            patience_counter = 0
            early_stopped = False
            early_stop_iteration = None

            pbar = tqdm(
                range(pga_its),
                desc="PGA",
                file=sys.stderr,
                position=0,
                # ascii=True,
                leave=True,
                # dynamic_ncols=True,
                # miniters=20,
            )
            lr_history = []

            eval_score = 0.0
            overlap = 0.0
            for iter_idx in pbar:
                # Check memory and evaluate periodically
                should_evaluate = (iter_idx % 10 == 0) or (
                    self.early_stopping_enabled
                    and iter_idx % self.early_stopping_eval_freq == 0
                )

                # Update progress bar less frequently to prevent newlines
                should_update_display = (iter_idx % 20 == 0) or should_evaluate

                if should_evaluate:
                    allocated, reserved = get_memory_stats()

                    # Get current learning rate
                    if optimizer is not None:
                        current_lr = optimizer.lr
                    else:
                        current_lr = scheduler.get_lr()

                    # Evaluate current performance
                    eval_score = self._evaluate_J_s(
                        s_current, EVAL_SAMPLE, prior_s
                    )  # Perform one batch eval

                    # Early stopping check
                    if (
                        self.early_stopping_enabled
                        and iter_idx % self.early_stopping_eval_freq == 0
                    ):
                        score_history.append(eval_score)
                        should_stop, patience_counter = self._check_early_stopping(
                            score_history, patience_counter
                        )

                        if should_stop:
                            early_stopped = True
                            early_stop_iteration = iter_idx
                            tqdm.write(f"\nEarly stopping triggered at iteration {iter_idx}")
                            tqdm.write(f"Best score achieved: {max(score_history):.6f}")
                            tqdm.write(f"Current score: {eval_score:.6f}")
                            tqdm.write(
                                f"No improvement for {self.early_stopping_patience} evaluations"
                            )
                            break

                    if prior_s.shape[0] > 0:
                        overlap = torch.max(
                            calc_overlap(s_current, prior_s, do_relu=True)
                        ).item()

                    lr_history.append(current_lr)

                # Update progress bar display only when needed
                if should_update_display and should_evaluate:
                    # Update progress bar with compact postfix to prevent line wrapping
                    postfix = {
                        "GPU": f"{allocated:.1f}G",
                        "J": f"{eval_score:.3f}",
                        "norm": f"{torch.linalg.norm(s_current, dim=-1).mean().item():.1f}",
                        "ovlp": f"{overlap:.2f}",
                        "lr": f"{current_lr:.1e}",
                    }

                    if self.early_stopping_enabled:
                        postfix[
                            "pat"
                        ] = f"{patience_counter}/{self.early_stopping_patience}"
                        if len(score_history) > 1:
                            best_score = (
                                max(score_history[:-1])
                                if len(score_history) > 1
                                else score_history[0]
                            )
                            improvement = eval_score - best_score
                            postfix["Δ"] = f"{improvement:+.3f}"

                    pbar.set_postfix(postfix)

                s_current = self._pga_iteration_enhanced(
                    s_current, optimizer, scheduler, iter_idx, prior_s
                )

            pbar.close()

            # Final evaluation if not done recently
            if not should_evaluate or early_stopped:
                eval_score = self._evaluate_J_s(
                    s_current, EVAL_SAMPLE, prior_s
                )  # Perform one batch eval

            # Get the overlaps of all the s_current vectors
            if prior_s.shape[0] > 0:
                overlap = calc_overlap(s_current, prior_s, do_relu=False)
                tqdm.write(
                    f"Overlap with prior s: {overlap.mean().item():.6f} (max: {overlap.max().item():.6f})"
                )

            # Print final results with early stopping info
            if early_stopped:
                tqdm.write(
                    f"Training completed early at iteration {early_stop_iteration}/{pga_its}"
                )
                tqdm.write(
                    f"Final evaluation J(s): {eval_score:.6f}, with norm {torch.linalg.norm(s_current, dim=-1)}"
                )
                tqdm.write(f"Best score during training: {max(score_history):.6f}")
            else:
                tqdm.write(f"Training completed full {pga_its} iterations")
                tqdm.write(
                    f"Final evaluation J(s): {eval_score:.6f}, with norm {torch.linalg.norm(s_current, dim=-1)}"
                )

            # Store results with metadata
            result_dict = {
                "vector": s_current.detach().cpu(),
                "final_score": eval_score,
                "lr_history": lr_history,
                "optimizer_type": self.optimizer_type,
                "scheduler_type": self.scheduler_type,
                "early_stopped": early_stopped,
                "early_stop_iteration": early_stop_iteration,
                "score_history": score_history,
                "total_iterations": early_stop_iteration if early_stopped else pga_its,
            }
            return result_dict

    def find_optimal_s_directions(self, do_sort=True) -> list[dict]:
        """Find optimal s directions with enhanced optimization."""
        print(f"Starting Enhanced PGA with {self.dict_size} candidates...")
        print(f"Using {self.optimizer_type} optimizer with {self.scheduler_type} scheduler")

        all_s_candidates = []

        if hasattr(self.model_segment, "base_model"):
            self.model_segment.base_model.eval()

        pga_its_offset = 0
        norm_idx = 0

        for i in range(self.dict_size):
            prior_s = (
                torch.stack([a["vector"] for a in all_s_candidates[:i]], dim=0).to(self.device)
                if i > 0
                else torch.zeros((0, self.kwise_coordinates, self.d_model), device=self.device)
            )

            if self.n_norm_discretization_steps > 0:
                offset = (self.norm_upper_bound - self.norm_lower_bound) * (norm_idx + 1) / self.n_norm_discretization_steps
                self.target_norm = self.norm_lower_bound + offset
                norm_idx = (norm_idx + 1) % self.n_norm_discretization_steps

            result_dict = self._find_optimal_s_single(prior_s, self.pga_iterations + pga_its_offset)
            pga_its_offset += 200
            all_s_candidates.append(result_dict)

        return self.final_evaluation_and_ranking([r["vector"] for r in all_s_candidates], do_sort=do_sort)

    @torch.no_grad()
    def _evaluate_J_s(
        self, s_current: torch.Tensor, num_samples: int, s_prior: torch.Tensor | None = None
    ) -> float:
        if s_prior is None:
            s_prior = torch.zeros((0, s_current.shape[0], s_current.shape[1]))
        batch_size = self.eval_batch_size
        sum_score = 0.0
        total_samples = 0
        for _ in range(math.ceil(num_samples / batch_size)):
            # Generate a batch of samples
            x_batch, tokens, pos_indices = self._generate_batch_samples(
                batch_size
            )
            x_sel = select_positions(x_batch, pos_indices)

            alpha = self._get_alpha(x_sel.shape[0])
            y_batch_embed = add_in_offset_vec(
                x_sel, s_current.to(x_sel.device), alpha, self.use_MICS
            )

            # Forward pass for original embeddings
            fx_logits = self.model_segment(x_sel, tokens, pos_indices)
            # Forward pass for perturbed embeddings
            fy_logits = self.model_segment(y_batch_embed, tokens, pos_indices)

            s = s_current.to(x_sel.device)
            objective = intervention_objective(
                fx_logits,
                fy_logits,
                x_sel,
                s_current,
                s_prior,
                overlap_penalty=0.0,
                target_norm=self.target_norm,
                norm_penalty=0.0,
            )

            total_samples += objective.shape[0]
            sum_score += objective.sum().item()
            del (
                x_batch,
                tokens,
                pos_indices,
                x_sel,
                y_batch_embed,
                fx_logits,
                fy_logits,
                objective,
            )

        return sum_score / total_samples

    def final_evaluation_and_ranking(
        self, all_s_candidates: list[torch.Tensor], do_sort=True
    ) -> list[dict]:
        """Final evaluation and ranking with memory optimization"""
        print("Final evaluation and ranking...")

        # Remove duplicates
        unique_candidates = all_s_candidates
        # unique_candidates = all_s_candidates
        print(f"Found {len(unique_candidates)} unique candidates")

        if not unique_candidates:
            return []

        # Evaluate candidates
        evaluated_candidates = []
        for i, s_cand in enumerate(
            tqdm(unique_candidates, desc="Final Eval", ncols=800, leave=False)
        ):
            with cuda_memory_manager():
                s_prior = (
                    torch.stack(unique_candidates[:i], dim=0).to(self.device)
                    if i > 0
                    else torch.zeros(
                        (0, self.kwise_coordinates, self.d_model), device=self.device
                    )
                )
                score = self._evaluate_J_s(
                    s_cand.to(self.device), self.num_x_samples_for_final_eval, s_prior
                )
                evaluated_candidates.append({"s": s_cand.cpu(), "score_J_s": score})
                print("Got score", score)

        # Sort and return top candidates
        if do_sort:
            evaluated_candidates.sort(key=lambda x: x["score_J_s"], reverse=True)

        # Transform each CSS direction into positive and negative feature pairs
        feature_pairs = []
        for i, candidate in enumerate(evaluated_candidates):
            base_feature_id = i
            s_vector = candidate["s"]
            score = candidate["score_J_s"]

            # Positive feature (original direction)
            positive_feature = {
                "s": s_vector,
                "score_J_s": score,
                "polarity": "positive",
                "base_feature_id": base_feature_id,
                "feature_id": f"{base_feature_id}_positive"
            }

            # Negative feature (negated direction)
            negative_feature = {
                "s": -s_vector,
                "score_J_s": score,  # Same score for both polarities
                "polarity": "negative",
                "base_feature_id": base_feature_id,
                "feature_id": f"{base_feature_id}_negative"
            }

            feature_pairs.extend([positive_feature, negative_feature])

        print(f"Generated {len(feature_pairs)} features ({len(evaluated_candidates)} pairs) with positive/negative polarities")
        return feature_pairs
