"""Learning rate schedulers and optimizer wrappers."""

import math

import torch
import bitsandbytes as bnb


class LearningRateScheduler:
    """Simple LR scheduler that computes learning rate from iteration number."""

    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def step(self, iteration: int) -> float:
        """Update and return current learning rate."""
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.current_lr


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay: lr = initial_lr * decay_rate^iteration"""

    def __init__(self, initial_lr: float, decay_rate: float = 0.995):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self, iteration: int) -> float:
        self.current_lr = self.initial_lr * (self.decay_rate**iteration)
        return self.current_lr


class CosineAnnealingScheduler(LearningRateScheduler):
    """Cosine annealing scheduler."""

    def __init__(self, initial_lr: float, total_iterations: int, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.total_iterations = total_iterations
        self.min_lr = min_lr

    def step(self, iteration: int) -> float:
        progress = iteration / self.total_iterations
        self.current_lr = (
            self.min_lr + (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        )
        return self.current_lr


class StepDecayScheduler(LearningRateScheduler):
    """Step decay: reduce lr by factor every step_size iterations."""

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.5):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, iteration: int) -> float:
        self.current_lr = self.initial_lr * (self.gamma ** (iteration // self.step_size))
        return self.current_lr


class LinearDecayScheduler(LearningRateScheduler):
    """Linear decay to minimum learning rate."""

    def __init__(self, initial_lr: float, total_iterations: int, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.total_iterations = total_iterations
        self.min_lr = min_lr

    def step(self, iteration: int) -> float:
        progress = min(iteration / self.total_iterations, 1.0)
        self.current_lr = self.initial_lr * (1 - progress) + self.min_lr * progress
        return self.current_lr


class SphericalOptimizerWrapper:
    """Wrapper adding sphere normalization to bitsandbytes AdamW8bit."""

    def __init__(
        self,
        params: torch.nn.Parameter,
        target_norm: float = -1.0,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.target_norm = target_norm
        self.params = params
        self.lr = lr
        self.optimizer = bnb.optim.AdamW8bit(
            [params], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    def step(self, params: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Perform optimizer step with sphere projection."""
        self.params.data = params.data
        self.params.grad = grad
        self.optimizer.step()

        params_new = self.params.data
        if self.target_norm > 0:
            params_norm = params_new.norm(dim=-1, keepdim=True)
            if torch.min(params_norm) > 1e-9:
                params_new = (params_new / params_norm) * self.target_norm
            else:
                raise ValueError("params_new norm too small, cannot normalize.")
        return params_new

    def reset(self):
        """Reset optimizer state."""
        self.optimizer.state.clear()
