import copy
import hashlib
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens
from datasets import IterableDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer


class TransformerDataHandler:
    """
    A memory-efficient data handler that generates batches of model activations on-the-fly.

    This class avoids pre-computing and storing the entire dataset's embeddings in memory.
    Instead, it processes a batch of text directly from the source dataset each time
    `get_batch` is called, making it suitable for very large datasets.

    Args:
        model_name (str): The name of the Hugging Face model to use (e.g., "gpt2-small").
        dataset (Dataset): The dataset loaded from Hugging Face Hub.
        dataset_text_column (str): The name of the column containing the text data.
        layer_cutoff (int): The model layer from which to extract activations.
        hook_type (str): The type of hook point to use (e.g., "hook_mlp_out", "hook_resid_post").
        max_seq_len (int): The maximum sequence length for tokenization.
        device (Optional[torch.device]): The device to run the model on. Defaults to CUDA if available.
    """

    def __init__(
        self,
        model_name: str,
        dataset: IterableDataset,
        dataset_text_column: str,
        layer_cutoff: int,
        hook_type: str,
        max_seq_len: int = 128,
        device: Optional[torch.device] = None,
        seed: int = 42,
        force_dataset_download: bool = False,
    ):
        self.model_name = model_name
        self.seed = seed
        self.dataset = dataset
        self.dataset_text_column = dataset_text_column
        self.layer_cutoff = layer_cutoff
        self.hook_type = hook_type
        self.max_seq_len = max_seq_len
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Dataset caching parameters
        self.force_dataset_download = force_dataset_download

        self.hook_point_name_for_x = f"blocks.{self.layer_cutoff}.{hook_type}"

        print(
            f"DataHandler: Initializing model {self.model_name} on device {self.device}..."
        )
        self.model: HookedTransformer = HookedTransformer.from_pretrained(
            self.model_name, device=self.device
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer

        if self.tokenizer.pad_token_id is None:
            print(
                f"DataHandler: Tokenizer does not have a pad_token_id. Using eos_token_id as default."
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._initialize_dataset()

    def _initialize_dataset(self):
        """
        Loads and prepares the dataset iterator, removing unused columns to prevent errors.
        """
        dataset: IterableDataset = self.dataset
        self.dataset = dataset.shuffle(seed=self.seed)

        # Create DataLoader with batch_size=1 for individual samples
        self.dataloader = DataLoader(self.dataset, batch_size=1)#, num_workers=0)
        self.dataset_iterator = iter(self.dataloader)

    def get_batch(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generates a batch of activations and corresponding tokens on-the-fly.
        If the dataset is exhausted, it will raise a StopIteration exception.

        Args:
            batch_size (int): The number of samples to include in the batch.

        Returns:
            A tuple containing:
            - x_batch (torch.Tensor): Activations from the target layer. Shape: (batch_size, max_seq_len, d_model)
            - x_batch_prior (torch.Tensor): Activations from the layer before the target. Shape: (batch_size, max_seq_len, d_model)
            - tokens_batch (torch.Tensor): The input token IDs. Shape: (batch_size, max_seq_len)
            Returns None if the batch_size is invalid or no data can be fetched.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        batch_texts = []
        # Fetch texts using DataLoader iterator
        for _ in range(batch_size):
            try:
                sample_batch = next(self.dataset_iterator)
                text_list = sample_batch[self.dataset_text_column]
                batch_texts.append(text_list[0])

            except StopIteration:
                # Re-raise the exception to signal the end of the dataset.
                # The caller can then handle re-initialization if needed.
                # re-init
                self.dataset_iterator = iter(self.dataset)
                print("DataHandler: Dataset exhausted, reinitializing iterator.")

        # If we couldn't gather any texts, stop.
        if not batch_texts:
            return None

        # --- Tokenization and Padding ---
        # The tokenizer expects a flat list of strings.
        tokens_on_device = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"].to(self.device)

        if self.layer_cutoff < 0:
            raise ValueError(
                "layer_cutoff must be >= 0 to extract hidden state activations."
            )

        # --- Model Inference and Activation Extraction ---
        try:
            with torch.no_grad():
                # Use run_with_cache to get intermediate activations
                _, cache = self.model.run_with_cache(
                    tokens_on_device,
                    names_filter=[
                        self.hook_point_name_for_x,
                    ],
                )
                x_batch = cache[self.hook_point_name_for_x]
        except KeyError as e:
            print(
                f"CRITICAL: Hook name {e} not found in model. Check model architecture and hook_type."
            )
            raise e

        return x_batch, tokens_on_device

    def get_norm_stats(
        self, samples: int, batch_size: int
    ) -> Tuple[float, float, float]:
        norms = []
        norms_max = []
        # Use ceiling division to ensure all samples are processed, including the last partial batch.
        num_batches = (samples + batch_size - 1) // batch_size

        for _ in tqdm(range(num_batches), desc="Calculating norm statistics",
                      file=sys.stderr, position=0, ascii=True,
                      dynamic_ncols=True, miniters=1):
            try:
                # 1. Get a batch to compute norms (only one call).
                batch_data = self.get_batch(batch_size)
            except StopIteration:
                print("Warning: Dataset exhausted before processing all requested samples for norm stats.")
                break

            # 2. Check if the data source is exhausted.
            if batch_data is None:
                break
            x, x_prior, toks = batch_data

            non_padding_mask = toks != self.tokenizer.pad_token_id

            # 4. Apply the mask to filter out padding from activations.
            x_filtered = x[non_padding_mask]
            x_prior_filtered = x_prior[non_padding_mask]

            # Ensure that we have some data left after removing padding.
            if x_filtered.shape[0] == 0:
                continue

            diffed = x_filtered - x_prior_filtered

            # Compute L2 norms along the feature dimension.
            batch_norms = diffed.norm(dim=-1)
            norms.append(batch_norms)
            norms_max.append(batch_norms.max().item())

        if not norms:
            raise RuntimeError(
                "DataHandler: No non-padding data was available to calculate norm statistics."
            )

        # Concatenate norms from all batches into a single tensor.
        all_norms = torch.cat(norms, dim=0)
        all_norms_max_mean = torch.tensor(norms_max, device=self.device)

        # Calculate statistics with clear naming.
        average_norm = all_norms.mean().item()
        std_dev_norm = all_norms.std().item()

        return average_norm, std_dev_norm, all_norms_max_mean.mean().item()

    @property
    def d_model(self) -> int:
        """Returns the model's hidden dimension size."""
        return self.model.cfg.d_model

    @property
    def vocab_size(self) -> int:
        """Returns the model's vocabulary size."""
        return self.model.cfg.d_vocab