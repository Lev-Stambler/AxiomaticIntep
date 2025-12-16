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


def robust_kl_divergence_batched(
    p_dist: torch.Tensor, q_dist: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Compute KL divergence D_KL(P || Q) for batched distributions.

    Args:
        p_dist: Distribution P, shape [C] or [B, C]
        q_dist: Distribution Q (reference), same shape as p_dist
        epsilon: Small constant to prevent log(0)

    Returns:
        KL divergence values, shape [] or [B]
    """
    log_q = torch.log(q_dist + epsilon)
    return F.kl_div(log_q, p_dist, reduction="none", log_target=False).sum(dim=-1)


def sample_multinomial(N: int, K: int, token_offset: int = 0, sorted: bool = True) -> torch.Tensor:
    """Sample K unique indices from [token_offset, N) uniformly.

    Args:
        N: Upper bound (exclusive)
        K: Number of samples
        token_offset: Lower bound offset (indices start from here)
        sorted: Whether to return sorted indices

    Returns:
        Tensor of K sampled indices
    """
    token_offset = max(0, token_offset)
    valid_range = N - token_offset
    probs = torch.ones(valid_range) / valid_range
    r = torch.multinomial(probs, K, replacement=False).to(torch.int64)
    if sorted:
        r, _ = r.sort(dim=-1)
    return r + token_offset
