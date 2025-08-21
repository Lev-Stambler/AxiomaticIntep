import time  # Not used in this class, but kept from original
from typing import Dict, List, Optional, Tuple, Union  # Added Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm  # Not used in this class, but kept from original
from transformer_lens import HookedTransformer
from ..core.data_handler import TransformerDataHandler


def _gather_pos(x: torch.Tensor, pos: torch.Tensor):
    outs = torch.zeros_like(x[:, 0, :]).to(x.device)
    for i in range(x.shape[0]):
        outs[i] = x[i, pos[i].item()]
    return outs

class IntervenableTransformerSegment:
    """
    Represents a segment of a HookedTransformer model, allowing for interventions
    by injecting custom embeddings at specified positions and streaming activations
    from subsequent layers. Hooks are prepared during initialization for efficiency.

    Supports activation patching for mechanistic interpretability experiments.
    """

    def __init__(
        self,
        base_model: HookedTransformer,
        layer_cutoff: int,
        data_handler: TransformerDataHandler,
        target_layers: List[int],
        include_final_logits=False,
        softmax_temper: float = 0.3,
        target_token_offset: int = 0,  # Offset for the target token position
    ):
        """
        Initializes the model segment and prepares hook functions.

        Args:
            base_model: The base HookedTransformer model.
            layer_cutoff: The layer index *at or after* which activations are streamed.
                          Intervention happens at a point determined by `data_handler.hook_point_name_for_x`,
                          which is typically before or at this `layer_cutoff`.
                          Activations will be streamed from `blocks.{layer_cutoff}.hook_resid_post` onwards.
            data_handler: Handler providing model-specific details like hook names.
            softmax_temper: Set to -1 to disable softmax on the output
        """
        self.base_model: HookedTransformer = base_model
        self.layer_cutoff: int = layer_cutoff
        self.softmax_temper: float = softmax_temper
        self.device: torch.device = base_model.cfg.device  # type: ignore
        self.d_model: int = base_model.cfg.d_model  # type: ignore
        self.data_handler: TransformerDataHandler = data_handler
        self.hook_type = data_handler.hook_type
        self.include_final_logits = include_final_logits
        self.target_token_offset: int = target_token_offset
        self.target_layers = target_layers

        if not (0 <= self.layer_cutoff <= self.base_model.cfg.n_layers):  # type: ignore
            raise ValueError(
                f"layer_cutoff ({self.layer_cutoff}) is out of bounds "
                f"[0, {self.base_model.cfg.n_layers}]"
            )  # type: ignore

        # Attributes to be set by __call__
        self.current_h_input: Optional[torch.Tensor] = None
        self.to_remove_dir : Optional[torch.Tensor] = None
        self.current_injection_pos: Optional[torch.Tensor] = None
        self.captured_outputs: Dict[Union[int, str], torch.Tensor] = {}

        # Prepare hook functions and list once
        self._fwd_hooks_list: List[Tuple[str, callable]] = self._prepare_hooks()
        self._remove_dir_hooks_list: List[Tuple[str, callable]] = self._prepare_remove_dir_hooks()
        print(
            "Hook for input is",
            self._fwd_hooks_list[0][0],
            "and output is",
            [f[0] for f in self._fwd_hooks_list[1:]],
            "with target offset",
            self.target_token_offset
        )

    def _hook_fn_remove_dir(
        self, activation_at_hook_point: torch.Tensor, hook
    ) -> torch.Tensor:
        """
        Hook function to inject embeddings. Uses self.current_h_input and self.current_injection_pos.
        """
        if self.to_remove_dir is None or self.current_injection_pos is None:
            # This should not happen if __call__ sets these correctly.
            raise RuntimeError(
                "to_remove_dir or current_injection_pos not set before injection hook."
            )

        act_batch_size, _, _ = activation_at_hook_point.shape

        dim = activation_at_hook_point.shape[-1]

        for b_idx in range(act_batch_size):
            if False:
                projed_down = (
                    torch.eye(dim).to(self.to_remove_dir.device) - torch.outer(self.to_remove_dir, self.to_remove_dir)
                ).to(activation_at_hook_point.device)
                activation_at_hook_point[
                    b_idx, self.current_injection_pos[b_idx]
                ] = activation_at_hook_point[
                    b_idx, self.current_injection_pos[b_idx]
                ] @ projed_down
            else: # Simply remove the direction
                activation_at_hook_point[
                    b_idx, self.current_injection_pos[b_idx]
                ] = activation_at_hook_point[
                    b_idx, self.current_injection_pos[b_idx]
                ] - self.to_remove_dir
        return activation_at_hook_point

    def _hook_fn_inject(
        self, activation_at_hook_point: torch.Tensor, hook
    ) -> torch.Tensor:
        """
        Hook function to inject embeddings. Uses self.current_h_input and self.current_injection_pos.
        """
        if self.current_h_input is None or self.current_injection_pos is None:
            # This should not happen if __call__ sets these correctly.
            raise RuntimeError(
                "current_h_input or current_injection_pos not set before injection hook."
            )

        act_batch_size, _, _ = activation_at_hook_point.shape

        if self.current_h_input.shape[0] != act_batch_size:
            raise ValueError(
                f"CRITICAL ERROR in _hook_fn_inject: Batch size mismatch: "
                f"current_h_input ({self.current_h_input.shape[0]}) vs activation ({act_batch_size})"
            )

        for b_idx in range(act_batch_size):
            activation_at_hook_point[
                b_idx, self.current_injection_pos[b_idx]
            ] = self.current_h_input[b_idx]
        return activation_at_hook_point

    def _get_target_pos(self, injection_pos: torch.Tensor) -> torch.Tensor:
        """Helper to determine target position based on offset."""
        if self.target_token_offset < 0:
            return -1 * torch.ones_like(injection_pos).to(injection_pos.device)
        return injection_pos + self.target_token_offset

    def _create_capture_hook_for_layer(self, current_layer_idx: int) -> callable:
        """
        Factory to create a capture hook for a specific layer.
        The returned hook uses self.current_injection_pos and self.captured_outputs.
        """

        def actual_capture_hook_fn(activation_at_hook_point: torch.Tensor, hook):
            if self.current_injection_pos is None:
                raise RuntimeError("current_injection_pos not set before capture hook.")

            # activation_at_hook_point shape: (batch_size, seq_len, d_model)
            # self.current_injection_pos shape: (batch_size, num_injections)
            target_pos = self._get_target_pos(self.current_injection_pos)
            # target_pos shape: (batch_size, num_injections)

            # We need to gather activations at target_pos
            #target_pos_expanded = target_pos.unsqueeze(-1).expand(-1, -1, d_model)

            captured_activations = _gather_pos(
                activation_at_hook_point, target_pos
            )
            # captured_activations shape: (batch_size, num_injections, d_model)

            self.captured_outputs[current_layer_idx] = captured_activations

            return activation_at_hook_point

        return actual_capture_hook_fn

    def _prepare_remove_dir_hooks(self) -> List[Tuple[str, callable]]:
        """
        Prepares the list of forward hooks during initialization.
        """
        hooks = []

        injection_hook_name = self.data_handler.hook_point_name_for_x
        hooks.append(
            (injection_hook_name, self._hook_fn_remove_dir)
        )  # Note: passing the method itself

        return hooks

    def _prepare_hooks(self) -> List[Tuple[str, callable]]:
        """
        Prepares the list of forward hooks during initialization.
        """
        fwd_hooks = []

        injection_hook_name = self.data_handler.hook_point_name_for_x
        fwd_hooks.append(
            (injection_hook_name, self._hook_fn_inject)
        )  # Note: passing the method itself

        for layer_idx_to_capture in range(self.layer_cutoff + 1, self.base_model.cfg.n_layers):  # type: ignore
            if layer_idx_to_capture in self.target_layers:
                capture_hook_name = f"blocks.{layer_idx_to_capture}.{self.hook_type}"
                fwd_hooks.append(
                    (
                        capture_hook_name,
                        self._create_capture_hook_for_layer(layer_idx_to_capture),
                    )
                )
        return fwd_hooks

    def call_with_removed_dir(
        self,
        orthogonal_dir: torch.Tensor,
        original_tokens_padded: torch.Tensor,
        injection_pos: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        """
        Runs the model segment, injecting y_embedding and streaming activations.

        Args:
            orthogonal_dir: (d_model)
            original_tokens_padded: (batch_size, seq_len)
            injection_pos: (batch_size, num_injections)

        Returns:
            Tuple of next token and output logits.
        """
        if orthogonal_dir.ndim != 1:
            raise ValueError(f"flatten_dir must be 2D.")
        d_model_emb = orthogonal_dir.shape[0]
        # --- Input Validation ---
        if d_model_emb != self.d_model:
            raise ValueError(f"y_embedding d_model mismatch.")
        max_seq_len = original_tokens_padded.shape[1]
        if torch.any(injection_pos < 0) or torch.any(injection_pos >= max_seq_len):
            raise ValueError(f"injection_pos out of bounds.")

        # --- Set instance variables for hooks to use ---
        self.to_remove_dir = orthogonal_dir.to(self.device)

        # injection_pos is used for indexing, typically fine on CPU or should match tensor device.
        # For simplicity, we ensure it's on the same device as activations if it's not already.
        # However, PyTorch often handles CPU index tensors with GPU data tensors correctly.
        # Let's keep it as is, assuming it's handled or user ensures it's on CPU.
        self.current_injection_pos = injection_pos

        tokens_for_attn_mask: torch.Tensor = original_tokens_padded.to(self.device)

        # --- Model Execution with Pre-defined Hooks ---
        with self.base_model.hooks(fwd_hooks=self._remove_dir_hooks_list):
            final_logits: torch.Tensor = self.base_model(tokens_for_attn_mask)[
                :, -1, :
            ]  # Last token
        out_logs = final_logits
        ids = torch.argmax(out_logs, dim=-1)
        next_token = self.data_handler.tokenizer.batch_decode(ids)
        return (next_token, out_logs)


    def __call__(
        self,
        y_embedding: torch.Tensor,
        original_tokens_padded: torch.Tensor,
        injection_pos: torch.Tensor,
    ) -> Dict[Union[int, str], torch.Tensor]:
        """
        Runs the model segment, injecting y_embedding and streaming activations.

        Args:
            y_embedding: (batch_size, num_s, d_model)
            original_tokens_padded: (batch_size, seq_len)
            injection_pos: (batch_size, num_injections)

        Returns:
            Dict mapping layer indices (int) or "final_logits" (str) to tensors
            of shape (batch_size, num_injections, d_model_or_d_vocab).
        """
        batch_size, num_S, d_model_emb = y_embedding.shape

        # --- Input Validation ---
        if d_model_emb != self.d_model:
            raise ValueError(f"y_embedding d_model mismatch.")
        if y_embedding.ndim != 3:
            raise ValueError(f"y_embedding must be 3D.")
        if not (
            injection_pos.ndim == 2
            and injection_pos.shape[0] == batch_size
            and injection_pos.shape[1] == num_S
        ):
            raise ValueError(f"injection_pos shape mismatch or ndim error.")
        max_seq_len = original_tokens_padded.shape[1]
        if torch.any(injection_pos < 0) or torch.any(injection_pos >= max_seq_len):
            raise ValueError(f"injection_pos out of bounds.")

        # --- Set instance variables for hooks to use ---
        self.current_h_input = y_embedding.to(self.device)
        # injection_pos is used for indexing, typically fine on CPU or should match tensor device.
        # For simplicity, we ensure it's on the same device as activations if it's not already.
        # However, PyTorch often handles CPU index tensors with GPU data tensors correctly.
        # Let's keep it as is, assuming it's handled or user ensures it's on CPU.
        self.current_injection_pos = injection_pos
        self.captured_outputs = {}  # Reset for this call

        tokens_for_attn_mask: torch.Tensor = original_tokens_padded.to(self.device)

        # --- Model Execution with Pre-defined Hooks ---
        with self.base_model.hooks(fwd_hooks=self._fwd_hooks_list):
            all_logits: torch.Tensor = self.base_model(tokens_for_attn_mask)

            # all_logits shape: (batch_size, seq_len, d_vocab)
            target_pos = self._get_target_pos(self.current_injection_pos)
            # target_pos shape: (batch_size, num_injections)

            # We need to gather logits at target_pos
            d_vocab = all_logits.shape[-1]
            #target_pos_expanded = target_pos.unsqueeze(-1).expand(-1, -1, d_vocab)

            final_logits = _gather_pos(all_logits, target_pos)
            # final_logits shape: (batch_size, num_injections, d_vocab)
        outs = {}
        if self.include_final_logits:
            if self.softmax_temper == -1:
                final_logits = final_logits
            else:
                final_logits = F.softmax(final_logits / self.softmax_temper, dim=-1)
            outs["final_logits"] = final_logits
        for i in self.target_layers:
            outs[i] = self.captured_outputs[i]

        return outs


    def call_with_intervention(
        self,
        intervention_dir: torch.Tensor,
        original_tokens: List[torch.Tensor],
        injection_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the model with and without an intervention, returning the logits for both.

        Args:
            intervention_dir: (d_model)
            original_tokens: List[(seq_len)]
            injection_pos: (batch_size, num_injections)

        Returns:
            Tuple of (original_logits, added_logits, subtracted_logits)
        """
        # Run without intervention
        orig_logs = []
        subed_logs = []
        added_logs = []
        for i, toks in enumerate(original_tokens):
            self.current_injection_pos = injection_pos[i:i+1]
            target_pos = self._get_target_pos(self.current_injection_pos)
            tokens_for_attn_mask: torch.Tensor = toks.to(self.device).unsqueeze(0)

            orig_logs.append( 
                _gather_pos(self.base_model(tokens_for_attn_mask), target_pos).squeeze()
            )
            # Run with intervention
            torch.cuda.empty_cache()
            with self.base_model.hooks(fwd_hooks=self._prepare_remove_dir_hooks()):
                # Add intervention via negative of subtraction
                self.to_remove_dir = -intervention_dir.to(self.device)
                all_logits_added = self.base_model(tokens_for_attn_mask)
                added_logs.append(_gather_pos(all_logits_added, target_pos).squeeze())
                # Subtract intervention
                self.to_remove_dir = intervention_dir.to(self.device)
                all_logits_subtracted = self.base_model(tokens_for_attn_mask)
                subed_logs.append(_gather_pos(
                    all_logits_subtracted, target_pos).squeeze()
                )

        return torch.stack(orig_logs), torch.stack(added_logs), torch.stack(subed_logs)
