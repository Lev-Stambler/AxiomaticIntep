import os
import pickle
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from .data_handler import TransformerDataHandler

# Fix sqlite3 module for ChromaDB compatibility
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Optional ChromaDB import for similarity search
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None


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

    This class uses ChromaDB for a fast, initial search and provides an
    optional, more precise aggregation-based re-ranking search.

    1. `search()`: Performs a fast, approximate search on flattened (K * d_model) vectors using ChromaDB.
    2. `search_with_aggregation()`: A two-stage search. First, it fetches a set of promising
       candidates from ChromaDB. Then, it performs exact pairwise similarity calculations on this
       smaller set, providing more granular and interpretable scores without crashing.
    """

    def __init__(
        self,
        data_handler: TransformerDataHandler,
        device: torch.device,
        d_model: int,
        scoring_type: str,
        collection_name: str = "activation_windows",
        chroma_db_path: str | None = None,
        save_activations: bool = True,
    ):
        """
        Initializes the searcher with ChromaDB.
        """
        self.data_handler = data_handler
        self.device = device
        self.d_model = d_model
        self.save_activations = save_activations

        print("Searcher got scoring type", scoring_type)
        if scoring_type not in ["cosine", "dot"]:
            raise ValueError("scoring_type must be 'cosine' or 'dot'.")
        self.scoring_type = scoring_type
        self.space = "cosine" if scoring_type == "cosine" else "ip"

        self.is_indexed: bool = False
        self.db_path = chroma_db_path

        # If a chroma_db_path is provided, use it to create a persistent client.
        if chroma_db_path:
            if not CHROMADB_AVAILABLE:
                raise ImportError(
                    "ChromaDB is required for persistent storage. Install with: pip install chromadb"
                )

            self.indexed_sample_path = f"{self.db_path}/indexed_sample_tokens.pkl"
            self.client = chromadb.PersistentClient(
                path=chroma_db_path,
            )
            # Check if the indexed_sample_tokens file exists (only if save_activations is enabled)
            try:
                if self.save_activations and os.path.exists(self.indexed_sample_path):
                    with open(self.indexed_sample_path, "rb") as f:
                        self.indexed_sample_tokens = pickle.load(f)
                        self.is_indexed = True if self.indexed_sample_tokens else False
                else:
                    self.indexed_sample_tokens = []
            except Exception as e:
                print(f"Error loading indexed sample tokens: {e}")
                self.indexed_sample_tokens = []

        else:
            if not CHROMADB_AVAILABLE:
                raise ImportError(
                    "ChromaDB is required for similarity search. Install with: pip install chromadb"
                )

            self.indexed_sample_tokens = []
            try:
                self.client = chromadb.Client()
            except Exception as e:
                raise RuntimeError(f"Failed to create in-memory ChromaDB client: {e}")

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": self.space}
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create or get collection '{collection_name}': {e}"
            )

    def _normalize_if_cosine(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Normalize vectors for cosine similarity calculation.
        This is used by the aggregation search for its precise calculations.
        """
        if self.scoring_type == "cosine":
            return torch.nn.functional.normalize(vectors, p=2, dim=-1)
        return vectors

    # _extract_activations_for_sample and _create_k_token_windows are unchanged
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
                if True:
                    return cache[hook_name].squeeze(0).cpu()
                prior_norm = (
                    cache[hook_name_prior].squeeze(0).norm(dim=-1, keepdim=True)
                )
                """
                This is supposed to give a sort of scale invariance on "how much is added" in
                """
                subed = cache[hook_name].squeeze(0) - cache[hook_name_prior].squeeze(
                    0
                )  # / prior_norm
                subed = subed.cpu()
                return subed
        except Exception as e:
            raise RuntimeError(f"Failed to extract activations: {e}")

    # build_index_incremental is unchanged
    def build_index_incremental(
        self, num_samples_to_index: int, batch_size: int = 256, start_fresh: bool = True
    ) -> None:
        """Build the ChromaDB search index by fetching and processing samples.

        Args:
            num_samples_to_index: Number of samples to index
            batch_size: Batch size for processing
            start_fresh: Whether to start fresh or continue from existing index

        Raises:
            RuntimeError: If indexing fails
        """
        print("Building ChromaDB index...")

        if start_fresh:
            if self.collection.count() > 0:
                self.client.delete_collection(name=self.collection.name)
                self.collection = self.client.create_collection(
                    name=self.collection.name, metadata={"hnsw:space": self.space}
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
                # ... (rest of the batch processing and indexing logic is identical to the previous ChromaDB version)
                remaining_samples = num_samples_to_index - total_samples_processed
                current_batch_size = min(batch_size, remaining_samples)
                data_handler = self.data_handler

                batch_data = data_handler.get_batch(current_batch_size)
                if batch_data is None:
                    break
                _, batch_tokens = batch_data
                num_in_this_batch = batch_tokens.size(0)
                newly_fetched_tokens = list(torch.unbind(batch_tokens.cpu(), dim=0))

                batch_embeddings, batch_metadatas, batch_ids = [], [], []
                for _sample_idx, tokens in enumerate(newly_fetched_tokens):
                    sample_idx = total_samples_processed + _sample_idx
                    pad_token_id = getattr(data_handler.tokenizer, "pad_token_id", None)
                    clean_tokens = (
                        tokens[tokens != pad_token_id]
                        if pad_token_id is not None
                        else tokens
                    )
                    activations = self._extract_activations_for_sample(
                        data_handler, clean_tokens
                    )
                    for i in range(activations.size(0)):
                        batch_ids.append(f"s{sample_idx}_t{i}")
                        batch_metadatas.append(
                            {"sample_idx": sample_idx, "token_pos": i}
                        )

                    batch_embeddings.append(activations.numpy().astype(np.float32))
                    self.indexed_sample_tokens.append(clean_tokens.cpu().numpy())
                if batch_embeddings:
                    conc = np.concatenate(batch_embeddings, axis=0)
                    self.collection.add(
                        embeddings=conc, metadatas=batch_metadatas, ids=batch_ids
                    )

                total_samples_processed += num_in_this_batch
                pbar.update(num_in_this_batch)
        self.is_indexed = True
        # Save the saved indexed sample tokens if a DB path is provided and save_activations is enabled
        if self.db_path and self.save_activations:
            with open(self.indexed_sample_path, "wb") as f:
                pickle.dump(self.indexed_sample_tokens, f)

        # print(f"\nSuccessfully indexed {self.collection.count()} token windows.")

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
            top_k: The final number of results to return.
            rerank_top_n: The number of candidates to fetch from Chroma for re-ranking.
                          A higher number increases accuracy at the cost of speed.
            aggregation: The aggregation method ('mean', 'sum', 'min', 'max', 'exp_sum').
        """
        assert self.is_indexed, "Index must be built before searching."

        simple_query_vec = query_vecs[
            0
        ]  # Simply choose the first vector for the initial query for now
        # STAGE 1: Get candidates from ChromaDB
        initial_results = self.collection.query(
            query_embeddings=simple_query_vec.unsqueeze(0).cpu().numpy().astype(np.float32),
            n_results=rerank_top_n,
            include=["metadatas", "embeddings"],
        )

        candidate_ids = initial_results["ids"][0]
        if not candidate_ids:
            return []
        metadatas = initial_results["metadatas"][0]
        sample_indices = [int(meta["sample_idx"]) for meta in metadatas]

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
        max_length = (
            self.data_handler.max_seq_len - 1
        )
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

        # Along the same token for different K, keep only the maximum score and set the rest to -inf

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
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
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
        return {
            "indexed": True,
            "total_windows": self.collection.count(),
            "window_size_K": self.K,
            "vector_dimension": self.K * self.d_model,
            "scoring_type": self.scoring_type,
            "num_data_handlers": len(self.data_handlers),
            "num_samples_referenced": len(self.indexed_sample_tokens),
            "collection_name": self.collection.name,
        }
