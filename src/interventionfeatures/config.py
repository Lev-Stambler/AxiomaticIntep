"""Hydra-based configuration for interventionfeatures."""

import hashlib
import json
import os
import random
from dataclasses import dataclass, field

import torch

try:
    import numpy as np
except ImportError:
    np = None


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducible results."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device() -> str:
    """Determine and set up the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_param_hash(params_dict: dict, prefix: str = "") -> str:
    """Generate a hash from parameters dictionary."""
    params_json = json.dumps(params_dict, sort_keys=True, default=str)
    hash_str = hashlib.md5(params_json.encode()).hexdigest()[:12]
    return f"{prefix}{hash_str}" if prefix else hash_str


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 1e-6
    eval_freq: int = 10


@dataclass
class ModelConfig:
    model_name: str = "EleutherAI/pythia-70m-deduped"
    model_name_sae: str = "pythia-70m-deduped"
    layer_cutoff: int = 2
    hook_type: str = "hook_resid_post"
    max_seq_len: int = 40
    target_norm: float = 10.0
    use_output_logits: bool = False
    target_token_offset: int = 0
    target_layers: list[int] | None = None
    n_norm_discretization_steps: int = 1


@dataclass
class TrainingConfig:
    pga_batch_size: int = 64
    eval_batch_size: int = 128
    pga_its: int = 512
    dict_size: int = 1
    learning_rate: float = 0.2
    k: int = 1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-4
    weight_decay: float = 1e-2
    sample_temp: float = 0.2
    norm_lower_bound: float = 0.5
    norm_upper_bound: float = 90.0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    inactive_mode: str = "soft"  # Options: "none", "hard", "soft"
    eps_check: float = 0.1  # Threshold for hard mode
    inactive_sigma: float = 0.1  # Sigma for soft Gaussian weighting


@dataclass
class DatabaseConfig:
    num_similar_to_find: int = 20
    index_size: int = 10000
    index_batch_size: int = 32
    scoring_type: str = "dot"
    chroma_db_path: str | None = None
    save_activations: bool = True
    use_MICS: bool = True


@dataclass
class LLMEndpointConfig:
    provider: str = "openai"
    model_name: str = "o3-mini"


@dataclass
class LLMConfig:
    explainer: LLMEndpointConfig = field(default_factory=LLMEndpointConfig)
    faithfulness: LLMEndpointConfig = field(default_factory=LLMEndpointConfig)
    positive_negative_explanations: bool = True


@dataclass
class DatasetConfig:
    dataset_name: str = "allenai/c4"
    dataset_config: str | None = None
    dataset_data_files: str | None = None
    dataset_split: str = "train"
    dataset_streaming: bool = True
    dataset_text_column: str = "text"
    force_dataset_download: bool = False


@dataclass
class RAVELConfig:
    """RAVEL benchmark configuration."""

    entity_types: list[str] = field(
        default_factory=lambda: ["cities", "nobel", "verbs", "objects", "occupations"]
    )
    attributes_per_entity: int = -1  # -1 means all attributes
    ravel_repo_path: str | None = None  # Path to cloned RAVEL repo
    use_cached_data: bool = True


@dataclass
class MIBConfig:
    """MIB (Mechanistic Interpretability Benchmark) configuration."""

    tasks: list[str] = field(
        default_factory=lambda: [
            "ioi",
            "arithmetic_add",
            "arithmetic_sub",
            "mcqa",
            "arc_easy",
            "ravel",
        ]
    )
    max_samples_per_task: int = -1  # -1 means all samples
    hf_cache_dir: str | None = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluations."""

    enabled_benchmarks: list[str] = field(default_factory=lambda: ["ravel", "mib"])
    directions_path: str = "directions.pkl"  # Relative to output_dir
    direction_indices: list[int] | None = None  # Which directions to use, None = all
    orthonormalize_featurizer: bool = True
    output_format: str = "json"  # json, csv, or pickle
    ravel: RAVELConfig = field(default_factory=RAVELConfig)
    mib: MIBConfig = field(default_factory=MIBConfig)


@dataclass
class Config:
    """Main configuration for interventionfeatures."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    seed: int = 62
    output_dir: str = "outputs"

    def get_dict(self) -> dict:
        """Return configuration as a dictionary."""
        from dataclasses import asdict

        return asdict(self)
