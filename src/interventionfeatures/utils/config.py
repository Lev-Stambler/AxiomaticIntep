#!/usr/bin/env python
# ruff: noqa: B904

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

try:
    import numpy as np
except ImportError:
    np = None

# TOML imports - use tomllib for Python 3.11+, tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# For writing TOML files
try:
    import tomli_w
except ImportError:
    tomli_w = None


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducible results.
    
    This function sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (including CUDA if available)
    - PyTorch deterministic algorithms for maximum reproducibility
    
    Args:
        seed: The random seed to use
        
    Raises:
        RuntimeError: If PyTorch deterministic algorithms cannot be enabled
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy seed if available
    if np is not None:
        np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set additional environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


class MainConfig:
    """Configuration class to hold all parameters for the FourierFeatures pipeline.

    This class manages all hyperparameters and configuration options for
    model training, data processing, and experimental settings.
    """

    def __init__(self) -> None:
        # Model parameters
        self.layer_cutoff: int = 2
        self.model_name: str = "EleutherAI/pythia-70m-deduped"
        self.seed: int = 62
        self.target_norm: float = 10.0
        self.model_name_sae: str = "pythia-70m-deduped"
        self.hook_type: str = "hook_resid_post"
        self.n_norm_discretization_steps = 1
        self.max_seq_len: int = 40
        self.use_output_logits: bool = False
        self.target_token_offset: int = 0
        self.target_layers: Optional[
            List[int]
        ] = None  # List[int] or None for auto-calculation

        # Training parameters
        self.pga_batch_size: int = 64
        self.eval_batch_size: int = 128
        self.pga_its: int = 128 * 4
        self.dict_size: int = 1
        self.learning_rate: float = 0.2
        self.k: int = 1

        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.eps: float = 1e-4
        self.weight_decay: float = 1e-2
        self.sample_temp: float = 0.2

        # Norm bounds for Ax optimization
        self.norm_lower_bound: float = 0.5
        self.norm_upper_bound: float = 90.0
        # Learning rate bounds for Ax optimization
        self.lr_lower_bound: float = 1e-4
        self.lr_upper_bound: float = 1.0
        self.num_trials: int = 5
        self.pga_its_for_ax: int = 150

        # Flags
        self.use_ax: bool = False

        # Language model for explainer (feature labeling)
        self.explainer_llm_provider: str = "openai"
        self.explainer_llm_model_name: str = "o3-mini"
        
        # Language model for faithfulness testing
        self.faithfulness_llm_provider: str = "openai"
        self.faithfulness_llm_model_name: str = "o3-mini"
        
        # Explanation mode
        self.positive_negative_explanations: bool = True

        # Early stopping parameters
        self.early_stopping_enabled: bool = False
        self.early_stopping_patience: int = 5
        self.early_stopping_min_delta: float = 1e-6
        self.early_stopping_eval_freq: int = 10
        self.early_stopping_patience_ax: int = 3
        self.early_stopping_min_delta_ax: float = 1e-4
        self.early_stopping_eval_freq_ax: int = 5

        # Database parameters
        self.num_similar_to_find: int = 20
        self.index_size: int = 10000
        self.index_batch_size: int = 32
        self.scoring_type: str = "dot"  # Similarity scoring type: "dot" or "cosine"
        self.chroma_db_path: Optional[str] = None  # Path for persistent ChromaDB storage
        self.save_activations: bool = True  # Whether to save activations to pickle files
        self.use_MICS: bool = True  # Whether to scale s_current by norms in add_in_offset_vec

        # Dataset parameters
        self.dataset_name: str = "allenai/c4"
        self.dataset_config: Optional[str] = None
        self.dataset_data_files: Optional[str] = None
        self.dataset_split: str = "train"
        self.dataset_streaming: bool = True
        self.dataset_text_column: str = "text"

        # Dataset caching parameters
        self.force_dataset_download: bool = False

        # Output directory
        self.output_dir: Optional[Path] = None

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "MainConfig":
        """Load configuration from a file (JSON or TOML auto-detected).

        Args:
            config_path: Path to the configuration file

        Returns:
            MainConfig instance with loaded parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Auto-detect file type based on extension
        if config_path.suffix.lower() == ".toml":
            return cls.from_toml_file(config_path)
        else:
            return cls.from_json_file(config_path)

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> "MainConfig":
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            MainConfig instance with loaded parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                params = json.load(f)

            config = cls()
            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: Unknown configuration parameter: {key}")

            # Validate scoring_type
            if config.scoring_type not in ["dot", "cosine"]:
                raise ValueError(f"scoring_type must be 'dot' or 'cosine', got: {config.scoring_type}")

            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    @classmethod
    def from_toml_file(cls, config_path: Union[str, Path]) -> "MainConfig":
        """Load configuration from a TOML file.

        Args:
            config_path: Path to the TOML configuration file

        Returns:
            MainConfig instance with loaded parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid or TOML support not available
        """
        if tomllib is None:
            raise ValueError(
                "TOML support not available. Install tomli for Python <3.11"
            )

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                params = tomllib.load(f)

            config = cls()
            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"Warning: Unknown configuration parameter: {key}")

            # Validate scoring_type
            if config.scoring_type not in ["dot", "cosine"]:
                raise ValueError(f"scoring_type must be 'dot' or 'cosine', got: {config.scoring_type}")

            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def get_dict(self) -> Dict[str, Any]:
        """Create a dictionary representation of the configuration.

        This method gathers all public, non-callable attributes of the instance
        into a dictionary, which is useful for serialization.

        Returns:
            A dictionary containing the configuration parameters.
        """
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a file (JSON or TOML auto-detected).

        Args:
            config_path: Path where to save the configuration

        Raises:
            OSError: If file cannot be written
        """
        config_path = Path(config_path)

        # Auto-detect file type based on extension
        if config_path.suffix.lower() == ".toml":
            self.save_to_toml_file(config_path)
        else:
            self.save_to_json_file(config_path)

    def save_to_json_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            config_path: Path where to save the configuration

        Raises:
            OSError: If file cannot be written
        """
        config_path = Path(config_path)

        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = self.get_dict()
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            raise OSError(f"Failed to save configuration to {config_path}: {e}")

    def save_to_toml_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a TOML file.

        Args:
            config_path: Path where to save the configuration

        Raises:
            OSError: If file cannot be written
            ValueError: If TOML writing support not available
        """
        if tomli_w is None:
            raise ValueError("TOML writing support not available. Install tomli-w")

        config_path = Path(config_path)

        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = self.get_dict()

            # Convert Path objects to strings and handle None values for TOML serialization
            for key, value in list(config_dict.items()):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                elif value is None:
                    # TOML doesn't support None values, so we remove them
                    # They'll be handled by the default values when loading
                    del config_dict[key]

            with open(config_path, "wb") as f:
                tomli_w.dump(config_dict, f)

            print(f"Configuration saved to: {config_path}")
        except Exception as e:
            raise OSError(f"Failed to save configuration to {config_path}: {e}")

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.layer_cutoff < 0:
            raise ValueError("layer_cutoff must be non-negative")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.pga_batch_size <= 0:
            raise ValueError("pga_batch_size must be positive")

        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.dict_size <= 0:
            raise ValueError("dict_size must be positive")

        if self.target_norm <= 0:
            raise ValueError("target_norm must be positive")

        if self.index_size <= 0:
            raise ValueError("index_size must be positive")

        if self.norm_lower_bound <= 0:
            raise ValueError("norm_lower_bound must be positive")

        if self.norm_upper_bound <= 0:
            raise ValueError("norm_upper_bound must be positive")

        if self.norm_lower_bound >= self.norm_upper_bound:
            raise ValueError("norm_lower_bound must be less than norm_upper_bound")

        if self.num_trials <= 0:
            raise ValueError("num_trials must be positive")

        if self.pga_its_for_ax <= 0:
            raise ValueError("pga_its_for_ax must be positive")

        print("Configuration validation passed")

    @property
    def llm_provider(self) -> str:
        """Backward compatibility property that returns explainer_llm_provider."""
        return self.explainer_llm_provider

    @llm_provider.setter
    def llm_provider(self, value: str) -> None:
        """Backward compatibility setter that sets explainer_llm_provider."""
        self.explainer_llm_provider = value

    @property
    def llm_model_name(self) -> str:
        """Backward compatibility property that returns explainer_llm_model_name."""
        return self.explainer_llm_model_name

    @llm_model_name.setter
    def llm_model_name(self, value: str) -> None:
        """Backward compatibility setter that sets explainer_llm_model_name."""
        self.explainer_llm_model_name = value

    def get_db_params_dict(self) -> Dict[str, Any]:
        """Get parameters relevant for database creation.

        Returns:
            Dictionary of database-relevant parameters
        """
        return {
            "model_name": self.model_name,
            "layer_cutoff": self.layer_cutoff,
            "hook_type": self.hook_type,
            "max_seq_len": self.max_seq_len,
            "index_size": self.index_size,
            "index_batch_size": self.index_batch_size,
            "scoring_type": self.scoring_type
        }

    def get_data_params_dict(self) -> Dict[str, Any]:
        """Get parameters relevant for data/results creation.

        Returns:
            Dictionary of data processing and training parameters
        """
        return {
            "model_name": self.model_name,
            "layer_cutoff": self.layer_cutoff,
            "hook_type": self.hook_type,
            "max_seq_len": self.max_seq_len,
            "use_output_logits": self.use_output_logits,
            "target_token_offset": self.target_token_offset,
            "target_layers": self.target_layers,
            "pga_batch_size": self.pga_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "pga_its": self.pga_its,
            "dict_size": self.dict_size,
            "learning_rate": self.learning_rate,
            "k": self.k,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "sample_temp": self.sample_temp,
            "num_similar_to_find": self.num_similar_to_find,
            "early_stopping_enabled": self.early_stopping_enabled,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_eval_freq": self.early_stopping_eval_freq,
            "norm_lower_bound": self.norm_lower_bound,
            "norm_upper_bound": self.norm_upper_bound,
            "num_trials": self.num_trials,
            "pga_its_for_ax": self.pga_its_for_ax,
        }


def generate_param_hash(params_dict: Dict[str, Any], prefix: str = "") -> str:
    """Generate a hash from parameters dictionary.

    Args:
        params_dict: Dictionary of parameters to hash
        prefix: Optional prefix for the hash string

    Returns:
        Hash string with optional prefix

    Raises:
        ValueError: If params_dict is empty or None
    """
    if not params_dict:
        raise ValueError("Parameters dictionary cannot be empty or None")

    try:
        # Create a sorted JSON string for consistent hashing
        params_json = json.dumps(params_dict, sort_keys=True, default=str)
        hash_obj = hashlib.md5(params_json.encode())
        hash_str = hash_obj.hexdigest()[:12]  # Use first 12 characters
        return f"{prefix}{hash_str}" if prefix else hash_str
    except Exception as e:
        raise ValueError(f"Failed to generate hash from parameters: {e}")


def create_directory_with_params(
    base_name: str, params_dict: Dict[str, Any], prefix: str = ""
) -> str:
    """Create directory with hash name and save parameters JSON.

    Args:
        base_name: Base name for the directory
        params_dict: Parameters to hash and save
        prefix: Optional prefix for the hash

    Returns:
        Path to the created directory

    Raises:
        OSError: If directory creation fails
        ValueError: If parameters are invalid
    """
    if not base_name.strip():
        raise ValueError("Base name cannot be empty")

    try:
        hash_name = generate_param_hash(params_dict, prefix)
        dir_path = f"./{base_name}-{hash_name}"

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save parameters to JSON file
        params_file = os.path.join(dir_path, "parameters.json")
        with open(params_file, "w") as f:
            json.dump(params_dict, f, indent=2, default=str)

        print(f"Created directory: {dir_path}")
        print(f"Parameters saved to: {params_file}")

        return dir_path
    except Exception as e:
        raise OSError(f"Failed to create directory with parameters: {e}")


def setup_device() -> str:
    """Determine and set up the best available device.

    Returns:
        Device string ('cuda', 'mps', or 'cpu')

    Raises:
        RuntimeError: If device setup fails
    """
    try:
        if torch.cuda.is_available():
            device = "cuda"
            # Test CUDA availability
            _ = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            # Test MPS availability
            _ = torch.ones(1, device="mps")
        else:
            device = "cpu"

        print(f"Using device: {device}")
        return device
    except Exception as e:
        raise RuntimeError(f"Failed to setup device: {e}")
