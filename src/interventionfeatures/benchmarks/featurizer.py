"""CSSFeaturizer - wraps CSS directions as pyvene-compatible featurizer."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import pyvene as pv

    PYVENE_AVAILABLE = True
except ImportError:
    PYVENE_AVAILABLE = False
    pv = None


class CSSFeaturizer(nn.Module):
    """
    Wraps CSS directions as a pyvene-compatible featurizer.

    CSS directions have shape (K, d_model) where K is typically 1 for single-coordinate
    interventions. The featurizer transforms activations by projecting onto CSS directions.

    For interchange interventions:
    1. Forward: Project activations onto CSS direction space
    2. Swap specified feature dimensions between base and source
    3. Inverse: Reconstruct activations from modified features

    Attributes:
        css_directions: Tensor of shape (num_directions, d_model) - orthonormal basis
        num_directions: Number of CSS directions
        d_model: Model hidden dimension
        dimensions_to_intervene: Which feature dimensions to swap during intervention
    """

    def __init__(
        self,
        css_directions: torch.Tensor,
        d_model: int,
        intervention_dims: list[int] | None = None,
        orthonormalize: bool = True,
    ):
        """
        Initialize CSSFeaturizer.

        Args:
            css_directions: CSS direction vectors. Shape can be:
                - (num_directions, K, d_model) - multiple directions with K coordinates
                - (K, d_model) - single direction
                - (num_directions, d_model) - multiple directions, K=1
            d_model: Model hidden dimension
            intervention_dims: Which feature indices to intervene on. Defaults to all.
            orthonormalize: Whether to orthonormalize the directions via Gram-Schmidt.
        """
        super().__init__()

        self.d_model = d_model

        # Normalize shape to (num_directions, d_model)
        if css_directions.dim() == 3:
            # (num_directions, K, d_model) -> (num_directions, d_model)
            # Flatten K dimension (typically K=1)
            css_directions = css_directions.squeeze(1)
        elif css_directions.dim() == 1:
            css_directions = css_directions.unsqueeze(0)

        self.num_directions = css_directions.shape[0]

        # Orthonormalize if requested to ensure bijective transformation
        if orthonormalize and self.num_directions > 1:
            css_directions = self._orthonormalize(css_directions)
        else:
            # Just normalize each direction
            css_directions = css_directions / css_directions.norm(dim=-1, keepdim=True)

        self.register_buffer("css_directions", css_directions)

        # Dimensions to intervene on (which CSS features to swap)
        self.dimensions_to_intervene = (
            intervention_dims if intervention_dims is not None else list(range(self.num_directions))
        )

    def _orthonormalize(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Create orthonormal basis from CSS directions using QR decomposition.

        Args:
            directions: Shape (num_directions, d_model)

        Returns:
            Orthonormal directions of same shape
        """
        # QR decomposition gives orthonormal basis
        Q, _ = torch.linalg.qr(directions.T)
        return Q.T[: self.num_directions]

    def forward_featurizer(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project activations onto CSS direction space.

        Args:
            activations: Shape (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            features: Shape (batch, [seq_len,] num_directions)
        """
        # Compute dot products with each CSS direction
        # css_directions: (num_directions, d_model)
        # activations: (..., d_model)
        features = torch.einsum("...d,nd->...n", activations, self.css_directions)
        return features

    def inverse_featurizer(self, features: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct activations from features, preserving orthogonal complement.

        Since CSS directions span a subspace, we preserve the component
        orthogonal to the CSS subspace from the original activations.

        Args:
            features: Shape (batch, [seq_len,] num_directions)
            residual: Original activations for orthogonal complement

        Returns:
            activations: Shape (batch, [seq_len,] d_model)
        """
        # Reconstruct the CSS subspace component from new features
        css_component = torch.einsum("...n,nd->...d", features, self.css_directions)

        # Compute the orthogonal component from original activations
        original_features = self.forward_featurizer(residual)
        original_css_component = torch.einsum(
            "...n,nd->...d", original_features, self.css_directions
        )
        orthogonal_component = residual - original_css_component

        # Combine: new CSS component + original orthogonal component
        return css_component + orthogonal_component

    def forward(
        self,
        base: torch.Tensor,
        source: torch.Tensor,
        subspaces: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Perform interchange intervention.

        Swaps specified feature dimensions from source into base activations.

        Args:
            base: Base activations to modify. Shape (batch, [seq_len,] d_model)
            source: Source activations to take features from. Same shape as base.
            subspaces: Which feature dimensions to swap. Defaults to self.dimensions_to_intervene

        Returns:
            Modified base activations with swapped features
        """
        dims = subspaces if subspaces is not None else self.dimensions_to_intervene

        # Get features for both
        base_features = self.forward_featurizer(base)
        source_features = self.forward_featurizer(source)

        # Swap specified dimensions
        intervened_features = base_features.clone()
        for dim in dims:
            intervened_features[..., dim] = source_features[..., dim]

        # Reconstruct with modified features
        return self.inverse_featurizer(intervened_features, base)

    def get_feature_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Get feature activation scores for each CSS direction.

        Useful for analyzing which features are active for given inputs.

        Args:
            activations: Shape (batch, seq_len, d_model)

        Returns:
            Feature scores: Shape (batch, seq_len, num_directions)
        """
        return self.forward_featurizer(activations)

    @classmethod
    def from_css_results(
        cls,
        css_results: list[dict],
        d_model: int,
        direction_indices: list[int] | None = None,
        use_positive_only: bool = False,
        orthonormalize: bool = True,
    ) -> CSSFeaturizer:
        """
        Create featurizer from CSSDirectionFinder output.

        Args:
            css_results: List of dicts with 's' key containing direction tensors
            d_model: Model hidden dimension
            direction_indices: Which directions to use (by index). None means all.
            use_positive_only: If True, only use positive polarity features
            orthonormalize: Whether to orthonormalize directions

        Returns:
            CSSFeaturizer instance
        """
        if direction_indices is None:
            results_to_use = css_results
        else:
            results_to_use = [css_results[i] for i in direction_indices]

        if use_positive_only:
            results_to_use = [r for r in results_to_use if r.get("polarity") == "positive"]

        # Extract direction tensors
        directions = torch.stack([r["s"] for r in results_to_use])

        return cls(
            css_directions=directions,
            d_model=d_model,
            orthonormalize=orthonormalize,
        )


if PYVENE_AVAILABLE:

    class CSSPyveneIntervention(pv.TrainableIntervention):
        """
        Pyvene-compatible intervention wrapper for CSSFeaturizer.

        This class allows CSS directions to be used directly with pyvene's
        intervention framework for RAVEL and MIB evaluations.
        """

        def __init__(
            self,
            css_featurizer: CSSFeaturizer,
            **kwargs,
        ):
            """
            Initialize pyvene intervention.

            Args:
                css_featurizer: CSSFeaturizer instance
                **kwargs: Additional args passed to pv.TrainableIntervention
            """
            super().__init__(**kwargs)
            self.css_featurizer = css_featurizer

        def forward(
            self,
            base: torch.Tensor,
            source: torch.Tensor,
            subspaces: list[int] | None = None,
        ) -> torch.Tensor:
            """Perform interchange intervention using CSS featurizer."""
            return self.css_featurizer(base, source, subspaces)

        @property
        def interchange_dim(self) -> int:
            """Return the dimension of the interchange space."""
            return self.css_featurizer.num_directions
