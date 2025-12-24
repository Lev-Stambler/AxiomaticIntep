"""Activation similarity search using Open-Puffer vector database.

This module provides vector similarity search for model activations using
the Open-Puffer high-performance vector database.
"""

import json
import os
import pickle
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from .data_handler import TransformerDataHandler
from .openpuffer_client import (
    OpenPufferServerManager,
    get_openpuffer_client,
)


def _find_optimal_ordered(
    tensor: torch.Tensor, score_function: str
) -> tuple[torch.Tensor, float]:
    """
    Finds the sequence of well-ordered indices that maximizes a scoring function and returns the score.
    This is a corrected and robust version.

    Args:
        tensor: A 2D tensor of shape (K, SEQ_LEN).
        score_function: The scoring function to use ('max', 'min', 'sum', 'mean', 'exp_sum').

    Returns:
        A tuple containing:
        - A 1D tensor of shape (K,) containing the optimal indices.
        - The optimal score as a float.
    """
    K, SEQ_LEN = tensor.shape
    if K > SEQ_LEN:
        raise ValueError(
            f"SEQ_LEN ({SEQ_LEN}) must be greater than or equal to K ({K})."
        )

    # Use float64 for precision in DP table
    tensor_d = tensor.double()
    dp = torch.zeros_like(tensor_d)
    paths = torch.zeros_like(tensor, dtype=torch.long)

    # Initialize the first row of the DP table
    dp[0] = tensor_d[0]

    # Fill DP and path tables for k > 0
    for k in range(1, K):
        # Determine running best values and indices from the previous row
        if score_function == "max":
            running_best_vals, running_best_indices = torch.cummin(dp[k - 1], dim=0)
        else:  # sum, mean, min, exp_sum
            running_best_vals, running_best_indices = torch.cummax(dp[k - 1], dim=0)

        # We need the best value from indices *less than* j.
        # We achieve this by shifting the cumulative best values.
        prev_best_vals = torch.roll(running_best_vals, shifts=1)
        prev_best_indices = torch.roll(running_best_indices, shifts=1)

        # Set the first element to an appropriate identity value
        if score_function == "max":
            prev_best_vals[0] = float("inf")
        else:
            prev_best_vals[0] = -float("inf")
        prev_best_indices[0] = -1

        # Update paths and DP table for the current row
        paths[k] = prev_best_indices
        if score_function in ["sum", "mean", "exp_sum"]:
            dp[k] = tensor_d[k] + prev_best_vals
        elif score_function == "min":
            dp[k] = torch.minimum(tensor_d[k], prev_best_vals)
        elif score_function == "max":
            dp[k] = torch.maximum(tensor_d[k], prev_best_vals)

        # *** THE CORRECTION ***
        # Invalidate scores for indices j that are too small for the current row k.
        # A valid path must have index i_k >= k.
        if score_function == "max":
            dp[k, :k] = float("inf")
        else:
            dp[k, :k] = -float("inf")

    # --- Find optimal score and backtrack for indices ---
    indices = torch.zeros(K, dtype=torch.long)

    # In the final row, we find the best score among the valid indices.
    # The mask applied in the loop already guarantees we are selecting from valid paths.
    if score_function == "max":
        optimal_score = dp[K - 1].min().item()
        indices[K - 1] = dp[K - 1].argmin()
    else:
        optimal_score = dp[K - 1].max().item()
        indices[K - 1] = dp[K - 1].argmax()

    # Backtrack from the last index to find the rest of the path
    for k in range(K - 2, -1, -1):
        indices[k] = paths[k + 1, indices[k + 1]]

    # For 'mean', the DP table calculated the sum, so we divide by K
    # For 'exp_sum', we apply the exponential to get e^(sum of scores)
    if score_function == "mean":
        optimal_score /= K
    elif score_function == "exp_sum":
        optimal_score = torch.exp(torch.tensor(optimal_score)).item()

    return indices, optimal_score


class ActivationSimSearcher:
    """
    Multi-token activation similarity searcher using a two-stage process.

    This class uses Open-Puffer for fast vector search and provides an
    optional, more precise aggregation-based re-ranking search.

    1. Initial search: Fast approximate search via Open-Puffer.
    2. `search_with_aggregation()`: Two-stage search with exact pairwise
       similarity calculations on promising candidates.
    """

    def __init__(
        self,
        data_handler: TransformerDataHandler,
        device: torch.device,
        d_model: int,
        scoring_type: str,
        collection_name: str = "activation_windows",
        db_path: str | None = None,
        save_activations: bool = True,
        openpuffer_host: str = "localhost",
        openpuffer_port: int = 8080,
        openpuffer_binary_path: str | None = None,
        openpuffer_data_dir: str | None = None,
    ):
        """
        Initializes the searcher with Open-Puffer.

        Args:
            data_handler: Handler for accessing model and data
            device: Torch device for computations
            d_model: Model dimension
            scoring_type: Either "cosine" or "dot" for similarity scoring
            collection_name: Name for the vector collection
            db_path: Path for persistent storage (tokens pickle file)
            save_activations: Whether to save indexed tokens
            openpuffer_host: Open-Puffer server host
            openpuffer_port: Open-Puffer server port
            openpuffer_binary_path: Path to puffer-server binary (for auto-start)
            openpuffer_data_dir: Data directory for Open-Puffer (for auto-start)
        """
        self.data_handler = data_handler
        self.device = device
        self.d_model = d_model
        self.save_activations = save_activations
        self.collection_name = collection_name

        print("Searcher got scoring type", scoring_type)
        if scoring_type not in ["cosine", "dot"]:
            raise ValueError("scoring_type must be 'cosine' or 'dot'.")
        self.scoring_type = scoring_type
        # Map to Open-Puffer metric names (Open-Puffer only supports "l2" or "cosine")
        # For dot product, we use cosine since it's most similar for normalized vectors
        self.metric = "cosine"

        self.is_indexed: bool = False
        self.db_path = db_path
        self.K = 1  # Will be updated during search

        # Set up persistence paths
        if db_path:
            self.indexed_sample_path = f"{db_path}/indexed_sample_tokens.pkl"
            self.metadata_path = f"{db_path}/collection_metadata.json"
            os.makedirs(db_path, exist_ok=True)
        else:
            self.indexed_sample_path = None
            self.metadata_path = None

        # Initialize Open-Puffer client
        self._server_manager: OpenPufferServerManager | None = None

        # Determine data directory for Open-Puffer
        puffer_data_dir = openpuffer_data_dir
        if puffer_data_dir is None and db_path:
            puffer_data_dir = f"{db_path}/puffer_data"

        try:
            self.client, self._server_manager = get_openpuffer_client(
                host=openpuffer_host,
                port=openpuffer_port,
                binary_path=openpuffer_binary_path,
                data_dir=puffer_data_dir,
                auto_start=bool(openpuffer_binary_path),
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to connect to Open-Puffer server: {e}\n"
                f"Either start the server manually with:\n"
                f"  ./puffer-server --bind-addr {openpuffer_host}:{openpuffer_port} --data-dir <path>\n"
                f"Or provide openpuffer_binary_path and openpuffer_data_dir for auto-start."
            ) from e

        # Try to load existing index
        self.indexed_sample_tokens: list = []
        if self.indexed_sample_path and os.path.exists(self.indexed_sample_path):
            try:
                with open(self.indexed_sample_path, "rb") as f:
                    self.indexed_sample_tokens = pickle.load(f)
                    if self.indexed_sample_tokens:
                        # Verify collection exists in Open-Puffer
                        if self.client.collection_exists(collection_name):
                            self.is_indexed = True
                            print(f"Loaded existing index with {len(self.indexed_sample_tokens)} samples")
                        else:
                            print("Collection not found in Open-Puffer, will rebuild index")
                            self.indexed_sample_tokens = []
            except Exception as e:
                print(f"Error loading indexed sample tokens: {e}")
                self.indexed_sample_tokens = []

        # Create collection if needed
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                name=collection_name,
                dimension=d_model,
                metric=self.metric,
            )
            print(f"Created Open-Puffer collection: {collection_name}")

    def _normalize_if_cosine(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Normalize vectors for cosine similarity calculation.
        This is used by the aggregation search for its precise calculations.
        """
        if self.scoring_type == "cosine":
            return torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors

    def _extract_activations_for_sample(
        self, data_handler: TransformerDataHandler, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Extract activations for a token sequence from a data handler."""
        try:
            with torch.no_grad():
                hook_name = data_handler.hook_point_name_for_x
                _, cache = data_handler.model.run_with_cache(
                    tokens.unsqueeze(0).to(self.device),
                    names_filter=[hook_name],
                )
                return cache[hook_name].squeeze(0).cpu()
        except Exception as e:
            raise RuntimeError(f"Failed to extract activations: {e}") from e

    def _extract_activations_batch(
        self, data_handler: TransformerDataHandler, batch_tokens: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extract activations for a batch of token sequences.

        Args:
            data_handler: Data handler with model
            batch_tokens: (batch_size, seq_len) tensor

        Returns:
            List of activation tensors (one per sample, padding removed)
        """
        try:
            with torch.no_grad():
                hook_name = data_handler.hook_point_name_for_x

                # Single forward pass for entire batch
                _, cache = data_handler.model.run_with_cache(
                    batch_tokens.to(self.device),
                    names_filter=[hook_name],
                )

                # Extract per-sample activations, removing padding
                activations_list = []
                pad_token_id = getattr(data_handler.tokenizer, "pad_token_id", None)

                for i in range(batch_tokens.size(0)):
                    tokens = batch_tokens[i]
                    acts = cache[hook_name][i].cpu()

                    # Remove padding tokens
                    if pad_token_id is not None:
                        mask = tokens.cpu() != pad_token_id
                        acts = acts[mask]

                    activations_list.append(acts)

                return activations_list
        except Exception as e:
            raise RuntimeError(f"Failed to extract batch activations: {e}") from e

    def build_index_incremental(
        self, num_samples_to_index: int, batch_size: int = 256, start_fresh: bool = True
    ) -> None:
        """Build the Open-Puffer search index by fetching and processing samples.

        Args:
            num_samples_to_index: Number of samples to index
            batch_size: Batch size for processing
            start_fresh: Whether to start fresh or continue from existing index

        Raises:
            RuntimeError: If indexing fails
        """
        print("Building Open-Puffer index...")

        if start_fresh:
            # Delete and recreate collection
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # Collection might not exist
            self.client.create_collection(
                name=self.collection_name,
                dimension=self.d_model,
                metric=self.metric,
            )
            self.is_indexed = False
            self.indexed_sample_tokens = []

        if not self.data_handler:
            raise ValueError("No data handler provided.")

        total_samples_processed = 0
        with tqdm(
            total=num_samples_to_index, desc="Indexing samples", unit="sample",
            file=sys.stderr, position=0, ascii=True,
            dynamic_ncols=True, miniters=10
        ) as pbar:
            while total_samples_processed < num_samples_to_index:
                remaining_samples = num_samples_to_index - total_samples_processed
                current_batch_size = min(batch_size, remaining_samples)
                data_handler = self.data_handler

                batch_data = data_handler.get_batch(current_batch_size)
                if batch_data is None:
                    break
                _, batch_tokens = batch_data
                num_in_this_batch = batch_tokens.size(0)

                # Extract activations in batch (optimized - single forward pass)
                batch_activations = self._extract_activations_batch(data_handler, batch_tokens)

                batch_embeddings, batch_metadatas, batch_ids = [], [], []
                for _sample_idx, (tokens, activations) in enumerate(
                    zip(torch.unbind(batch_tokens.cpu(), dim=0), batch_activations, strict=True)
                ):
                    sample_idx = total_samples_processed + _sample_idx

                    # Clean tokens (remove padding)
                    pad_token_id = getattr(data_handler.tokenizer, "pad_token_id", None)
                    clean_tokens = (
                        tokens[tokens != pad_token_id]
                        if pad_token_id is not None
                        else tokens
                    )

                    # Activations already computed in batch - just use them
                    for i in range(activations.size(0)):
                        batch_ids.append(f"s{sample_idx}_t{i}")
                        batch_metadatas.append(
                            {"sample_idx": sample_idx, "token_pos": i}
                        )

                    batch_embeddings.append(activations.numpy().astype(np.float32))
                    self.indexed_sample_tokens.append(clean_tokens.cpu().numpy())

                if batch_embeddings:
                    conc = np.concatenate(batch_embeddings, axis=0)
                    # Insert into Open-Puffer
                    self.client.insert_points(
                        collection_name=self.collection_name,
                        ids=batch_ids,
                        vectors=conc,
                        payloads=batch_metadatas,
                    )

                total_samples_processed += num_in_this_batch
                pbar.update(num_in_this_batch)

        self.is_indexed = True

        # Save indexed sample tokens if a DB path is provided
        if self.db_path and self.save_activations:
            with open(self.indexed_sample_path, "wb") as f:
                pickle.dump(self.indexed_sample_tokens, f)

            # Save metadata for recovery verification
            metadata = {
                "collection_name": self.collection_name,
                "d_model": self.d_model,
                "scoring_type": self.scoring_type,
                "num_samples": len(self.indexed_sample_tokens),
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f)

        print(f"Successfully indexed {len(self.indexed_sample_tokens)} samples.")

    def search_with_aggregation(
        self,
        query_vecs: torch.Tensor,
        use_cosine_scoring: bool,
        top_k: int = 10,
        rerank_top_n: int = 100,
        aggregation: str = "min",
    ) -> tuple[list[dict], torch.Tensor]:
        """
        Performs a two-stage search with aggregation-based re-ranking.

        Args:
            query_vecs: The (K, d_model) query tensor.
            use_cosine_scoring: Whether to use cosine similarity for scoring.
            top_k: The final number of results to return.
            rerank_top_n: The number of candidates to fetch for re-ranking.
                          A higher number increases accuracy at the cost of speed.
            aggregation: The aggregation method ('mean', 'sum', 'min', 'max', 'exp_sum').

        Returns:
            Tuple of (results list, scores_by_vec tensor)
        """
        assert self.is_indexed, "Index must be built before searching."

        simple_query_vec = query_vecs[0]  # Use first vector for initial query

        # STAGE 1: Get candidates from Open-Puffer
        initial_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=simple_query_vec.cpu().numpy().astype(np.float32),
            top_k=rerank_top_n,
            include_vectors=False,
            include_payload=True,
        )

        # Extract candidate information from results
        results_list = initial_results.get("results", [])
        if not results_list:
            return [], torch.tensor([])

        metadatas = [r.get("payload", {}) for r in results_list]
        sample_indices = [int(meta.get("sample_idx", 0)) for meta in metadatas]

        # Remove duplicates while preserving order
        seen_sample_indices = set()
        unique_sample_indices = []
        unique_metadatas = []
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx not in seen_sample_indices:
                seen_sample_indices.add(sample_idx)
                unique_sample_indices.append(sample_idx)
                unique_metadatas.append(metadatas[i])

        # Get all the embeddings in the sequence associated with the unique candidate sample indices
        indexed_sample_tokens = [
            self.indexed_sample_tokens[s] for s in unique_sample_indices
        ]
        embeddings_per_tok = [
            self._extract_activations_for_sample(
                self.data_handler, torch.tensor(tokens)
            )
            for tokens in indexed_sample_tokens
        ]

        # Convert to a single tensor of shape (N, K, d_model)
        self.K = query_vecs.size(0)
        self.d_model = query_vecs.size(1)

        # Pad the embeddings to ensure they are all the same length
        max_length = self.data_handler.max_seq_len - 1
        for i in range(len(embeddings_per_tok)):
            if embeddings_per_tok[i].size(0) < max_length:
                padding = torch.zeros(
                    max_length - embeddings_per_tok[i].size(0), self.d_model
                )
                embeddings_per_tok[i] = torch.cat(
                    (embeddings_per_tok[i], padding), dim=0
                )
            elif embeddings_per_tok[i].size(0) > max_length:
                embeddings_per_tok[i] = embeddings_per_tok[i][:max_length, :]

        embeddings_per_tok = torch.stack(embeddings_per_tok, dim=0).to(
            self.device
        )  # Shape: (BS, SEQ_LEN, d_model)

        score_by_vec = []
        for vec in query_vecs:
            if self.scoring_type == "cosine":
                vec = self._normalize_if_cosine(vec)
            scores = torch.sum(
                embeddings_per_tok * vec.unsqueeze(0).unsqueeze(0).to(embeddings_per_tok.device),
                dim=-1,
            ) if not use_cosine_scoring else torch.nn.functional.cosine_similarity(
                embeddings_per_tok, vec.unsqueeze(0).unsqueeze(0).to(embeddings_per_tok.device),
                dim=-1
            )
            score_by_vec.append(scores)
        score_by_vec = torch.stack(score_by_vec, dim=0)  # Shape: (K, N, SEQ_LEN)

        N = score_by_vec.size(1)  # Number of candidate sequences
        final_scores = torch.zeros(N, device=self.device)
        top_indices_for_all = torch.zeros(
            (N, self.K), dtype=torch.long, device=self.device
        )

        for i in range(N):
            e = score_by_vec[:, i, :]
            # Get the actual sequence length for this sample (before padding)
            actual_seq_len = len(indexed_sample_tokens[i])
            # Mask out positions beyond the actual sequence length
            e_masked = e.clone()
            if actual_seq_len < e.size(1):
                if aggregation == "max":
                    e_masked[:, actual_seq_len:] = float("inf")
                else:
                    e_masked[:, actual_seq_len:] = -float("inf")

            indices_per_item, score = _find_optimal_ordered(
                e_masked, score_function=aggregation
            )
            top_indices_for_all[i] = indices_per_item
            final_scores[i] = score

        # Return the top K results based on the final scores
        top_scores, top_indices = torch.topk(
            final_scores, k=min(top_k, final_scores.size(0))
        )
        top_score_by_vec = score_by_vec[:, top_indices, :].transpose(
            0, 1
        )  # Shape: (top_k, K, SEQ_LEN)

        results = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist(), strict=True):
            metadata = unique_metadatas[idx]
            results.append(
                {
                    "score": score,
                    "max_indices": top_indices_for_all[idx].cpu().tolist(),
                    **metadata,
                }
            )

        return results, top_score_by_vec

    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        if not self.is_indexed:
            return {"indexed": False}

        try:
            stats = self.client.get_collection_stats(self.collection_name)
            point_count = stats.get("point_count", 0)
        except Exception:
            point_count = 0

        return {
            "indexed": True,
            "total_windows": point_count,
            "window_size_K": getattr(self, "K", 1),
            "vector_dimension": self.d_model,
            "scoring_type": self.scoring_type,
            "num_samples_referenced": len(self.indexed_sample_tokens),
            "collection_name": self.collection_name,
        }

    def count(self) -> int:
        """Get the number of vectors in the index."""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats.get("point_count", 0)
        except Exception:
            return 0

    def __del__(self):
        """Cleanup server manager on destruction."""
        if hasattr(self, "_server_manager") and self._server_manager:
            self._server_manager.stop()
