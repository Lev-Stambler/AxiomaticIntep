import math

import torch
import torch.nn.functional as F


def find_orthogonal_vector(s_unit: torch.Tensor) -> torch.Tensor:
    """Finds a batch of unit vectors `v` orthogonal to input `s_unit`."""
    num_vecs, dim = s_unit.shape
    if dim == 1:
        raise ValueError("Cannot find an orthogonal vector in 1D space.")

    temp_v = torch.zeros_like(s_unit)
    temp_v[:, 0] = 1.0

    dot_prod = (temp_v * s_unit).sum(dim=-1)
    parallel_mask = torch.abs(dot_prod) > 1.0 - 1e-6

    if torch.any(parallel_mask):
        temp_v[parallel_mask, 0] = 0.0
        temp_v[parallel_mask, 1] = 1.0

    v = temp_v - (temp_v * s_unit).sum(dim=-1, keepdim=True) * s_unit

    return F.normalize(v, p=2, dim=-1)


def minimally_change_x(x: torch.Tensor, s: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Minimally changes x to x' with same norm and target cosine similarity to s.
    x shape: (B, K, D), s shape: (K, D). Operations are on dimension D.
    """
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [-1, 1]")

    x, s = x.float(), s.float()

    # Expand s to be broadcastable with x
    s_expanded = s.unsqueeze(0).expand_as(x)

    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    s_norm = torch.linalg.norm(s_expanded, dim=-1, keepdim=True)

    if torch.any(x_norm < 1e-8):
        return torch.zeros_like(x)
    if torch.any(s_norm < 1e-8):
        raise ValueError("Vector s cannot contain a zero vector.")

    s_unit = F.normalize(s_expanded, p=2, dim=-1)

    x_dot_s_unit = (x * s_unit).sum(dim=-1, keepdim=True)
    x_perp_s = x - x_dot_s_unit * s_unit
    x_perp_s_norm = torch.linalg.norm(x_perp_s, dim=-1, keepdim=True)

    v = torch.zeros_like(x)

    collinear_mask = (x_perp_s_norm < 1e-6).squeeze(-1)
    non_collinear_mask = ~collinear_mask

    if torch.any(non_collinear_mask):
        # Masking flattens the B, K dims, so normalize over the last dim
        v[non_collinear_mask] = F.normalize(x_perp_s[non_collinear_mask], p=2, dim=-1)

    if torch.any(collinear_mask):
        v[collinear_mask] = find_orthogonal_vector(s_unit[collinear_mask])

    cos_comp = alpha * s_unit
    sin_comp = math.sqrt(1.0 - alpha**2) * v

    x_prime = x_norm * (cos_comp + sin_comp)

    return x_prime


def robust_kl_divergence_batched(p_dist, q_dist, epsilon=1e-6):
    """
    Calculates KL divergence D_KL(P || Q) robustly for batches of distributions.
    P is represented by p_dist, and Q (the reference distribution) by q_dist.

    The formula for KL divergence is: sum_i ( P_i * (log P_i - log Q_i) ).
    PyTorch's F.kl_div(input, target, reduction='none', log_target=False) computes
    element-wise: target_i * (log target_i - input_i).
    To compute D_KL(P || Q):
    - 'target' should be P (p_dist).
    - 'input' should be log Q (log(q_dist + epsilon)).
    The result is then summed over the distribution dimension.

    Args:
        p_dist (torch.Tensor): The first probability distribution(s) (P).
                                 Shape: [C] for a single distribution, or [B, C] for a batch.
                                 B is batch size, C is number of classes/categories.
        q_dist (torch.Tensor): The second probability distribution(s) (Q, the reference).
                                 Must have the same shape as p_dist.
        epsilon (float): A small constant added to q_dist before taking the
                         logarithm to prevent log(0) issues, which would make
                         KL divergence infinite if p_dist_i > 0 and q_dist_i = 0.

    Returns:
        torch.Tensor: The KL divergence for each distribution in the batch.
                      Shape: [] (scalar tensor) if input is 1D [C].
                      Shape: [B] if input is 2D [B, C].
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(p_dist, torch.Tensor):
        p_dist = torch.tensor(p_dist, dtype=torch.float32)
    if not isinstance(q_dist, torch.Tensor):
        q_dist = torch.tensor(q_dist, dtype=torch.float32)

    # Validate shapes
    # if p_dist.shape != q_dist.shape:
    #    raise ValueError("Input distributions p_dist and q_dist must have the same shape.")
    if p_dist.dim() == 0:
        raise ValueError("Input distributions must be at least 1D.")

    # Ensure inputs are non-negative (they should be probabilities)
    if torch.any(p_dist < 0) or torch.any(q_dist < 0):
        # This is a basic check; for actual probability distributions,
        # elements must also sum to 1.
        print(
            "Warning: Input distributions contain negative values. KL divergence is typically for non-negative distributions."
        )

    log_q_dist = torch.log(q_dist + epsilon)
    element_wise_kl = F.kl_div(log_q_dist, p_dist, reduction="none", log_target=False)
    kl_div_values = torch.sum(element_wise_kl, dim=-1)

    return kl_div_values


def sample_multinomial(N, K, sorted=True):
    """
    We want to have some order, so we have samples which are sorted.
    """
    # Create uniform probabilities
    probs = torch.ones(N) / N
    r = torch.multinomial(probs, K, replacement=False).type(torch.int64)
    if sorted:
        r, _ = r.sort(dim=-1)  # Sort the last dimensionin ascending order
    return r
