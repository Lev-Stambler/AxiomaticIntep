import math
from typing import Dict, List, Optional, Tuple

import torch


# Learning Rate Schedulers
class LearningRateScheduler:
    """Base class for learning rate schedulers"""

    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def step(self, iteration: int) -> float:
        """Update and return current learning rate"""
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.current_lr


class ExponentialDecayScheduler(LearningRateScheduler):
    """Exponential decay: lr = initial_lr * decay_rate^iteration"""

    def __init__(self, initial_lr: float, decay_rate: float = 0.99):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self, iteration: int) -> float:
        self.current_lr = self.initial_lr * (self.decay_rate**iteration)
        return self.current_lr


class CosineAnnealingScheduler(LearningRateScheduler):
    """Cosine annealing scheduler"""

    def __init__(self, initial_lr: float, total_iterations: int, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.total_iterations = total_iterations
        self.min_lr = min_lr

    def step(self, iteration: int) -> float:
        progress = iteration / self.total_iterations
        self.current_lr = (
            self.min_lr
            + (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        )
        return self.current_lr


class StepDecayScheduler(LearningRateScheduler):
    """Step decay: reduce lr by factor every step_size iterations"""

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.5):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, iteration: int) -> float:
        self.current_lr = self.initial_lr * (
            self.gamma ** (iteration // self.step_size)
        )
        return self.current_lr


class LinearDecayScheduler(LearningRateScheduler):
    """Linear decay to minimum learning rate"""

    def __init__(self, initial_lr: float, total_iterations: int, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.total_iterations = total_iterations
        self.min_lr = min_lr

    def step(self, iteration: int) -> float:
        progress = min(iteration / self.total_iterations, 1.0)
        self.current_lr = self.initial_lr * (1 - progress) + self.min_lr * progress
        return self.current_lr


class SphericalAdamW:
    """AdamW optimizer adapted for optimization on the unit sphere"""

    def __init__(
        self,
        target_norm: float = -1.0,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.lr = lr
        self.target_norm = target_norm
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State variables
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0  # Time step

    def step(self, params: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Perform one optimization step"""
        self.t += 1

        # Initialize moment estimates if first step
        if self.m is None:
            self.m = torch.zeros_like(grad)
            self.v = torch.zeros_like(grad)

        # Add weight decay (L2 regularization)
        grad_with_decay = grad + self.weight_decay * params

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_with_decay

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_with_decay**2

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        params_new = params + update #* torch.norm(params, dim=-1).mean()
        if self.target_norm > 0:
            params_norm = params_new.norm(dim=-1)
            if torch.min(params_norm) > 1e-9:
               params_new = (params_new / params_norm) * self.target_norm
            else:
               # Fallback to random direction if norm is too small
               raise ValueError("params_new norm is too small, cannot normalize. There may be a bug here!")

        return params_new

    def reset(self):
        """Reset optimizer state"""
        self.m = None
        self.v = None
        self.t = 0
