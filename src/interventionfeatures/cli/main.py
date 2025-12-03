#!/usr/bin/env python3
"""CLI for interventionfeatures - CSS direction finding for mechanistic interpretability."""

import json
import os
import pickle
import sys
from pathlib import Path

# Fix sqlite3 for ChromaDB compatibility
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import hydra
from omegaconf import DictConfig, OmegaConf


def setup_environment():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def setup_pipeline(cfg: DictConfig):
    """Initialize pipeline components from config."""
    from datasets import load_dataset

    from ..config import set_global_seed, setup_device
    from ..core.data_handler import TransformerDataHandler
    from ..core.model import IntervenableTransformerSegment

    set_global_seed(cfg.seed)
    device = setup_device()

    dataset = load_dataset(
        cfg.dataset.dataset_name,
        cfg.dataset.dataset_config,
        data_files=cfg.dataset.dataset_data_files if cfg.dataset.dataset_data_files else None,
        split=cfg.dataset.dataset_split,
        streaming=cfg.dataset.dataset_streaming,
        token=os.environ.get("HUGGINGFACE_API_KEY"),
    )

    data_handler = TransformerDataHandler(
        dataset=dataset,
        dataset_text_column=cfg.dataset.dataset_text_column,
        model_name=cfg.model.model_name,
        layer_cutoff=cfg.model.layer_cutoff,
        device=device,
        max_seq_len=cfg.model.max_seq_len,
        hook_type=cfg.model.hook_type,
        seed=cfg.seed,
        force_dataset_download=cfg.dataset.force_dataset_download,
    )

    target_layers = (
        list(cfg.model.target_layers) if cfg.model.target_layers
        else [cfg.model.layer_cutoff + 1]
    )

    model_segment = IntervenableTransformerSegment(
        base_model=data_handler.model,
        data_handler=data_handler,
        layer_cutoff=cfg.model.layer_cutoff,
        target_layers=target_layers,
        include_final_logits=cfg.model.use_output_logits,
        target_token_offset=cfg.model.target_token_offset,
    )

    return data_handler, model_segment, device


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Run the CSS direction finding pipeline."""
    setup_environment()

    print("=" * 60)
    print("InterventionFeatures - CSS Direction Finding")
    print("=" * 60)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"Config saved to: {config_path}")

    # Setup pipeline
    print("\nInitializing pipeline...")
    data_handler, model_segment, device = setup_pipeline(cfg)
    print(f"Using device: {device}")

    # Find CSS directions
    from ..core.css import CSSDirectionFinder

    print("\nFinding CSS directions...")
    css_finder = CSSDirectionFinder.from_config(cfg, data_handler, model_segment)
    css_results = css_finder.find_optimal_s_directions()

    # Save results
    results_path = output_dir / "directions.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(css_results, f)
    print(f"Results saved to: {results_path}")

    # Build search index
    from ..core.search import ActivationSimSearcher

    print("\nBuilding similarity search index...")
    searcher = ActivationSimSearcher(
        data_handler=data_handler,
        d_model=data_handler.d_model,
        device=device,
        scoring_type=cfg.database.scoring_type,
        chroma_db_path=str(output_dir / "searcher_db"),
        save_activations=cfg.database.save_activations,
    )

    if not searcher.is_indexed:
        searcher.build_index_incremental(
            cfg.database.index_size,
            cfg.database.index_batch_size
        )

    print("\nPipeline complete!")
    print(f"Output directory: {output_dir}")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def explain(cfg: DictConfig) -> None:
    """Generate explanations for CSS directions."""
    setup_environment()

    output_dir = Path(cfg.output_dir)
    directions_path = output_dir / "directions.pkl"

    if not directions_path.exists():
        print(f"Error: directions.pkl not found at {directions_path}")
        print("Run the 'run' command first to generate CSS directions.")
        sys.exit(1)

    with open(directions_path, "rb") as f:
        css_results = pickle.load(f)

    from ..analysis.explainer import FeatureExplainer

    print(f"\nGenerating explanations for {len(css_results)} directions...")

    explainer = FeatureExplainer(
        llm_provider=cfg.llm.explainer.provider,
        llm_model_name=cfg.llm.explainer.model_name,
    )

    explanations = []
    for i, result in enumerate(css_results):
        print(f"  Direction {i+1}/{len(css_results)}...")
        # This is a placeholder - actual implementation depends on FeatureExplainer API
        explanations.append({"direction_id": i, "explanation": "TODO"})

    explanations_path = output_dir / "explanations.json"
    with open(explanations_path, "w") as f:
        json.dump(explanations, f, indent=2)

    print(f"Explanations saved to: {explanations_path}")


def print_config():
    """Print the default configuration."""
    from dataclasses import asdict

    from ..config import Config

    cfg = Config()
    print(OmegaConf.to_yaml(OmegaConf.create(asdict(cfg))))


def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: interventionfeatures <command> [options]")
        print("")
        print("Commands:")
        print("  run      Run the CSS direction finding pipeline")
        print("  explain  Generate explanations for CSS directions")
        print("  config   Print the default configuration")
        print("")
        print("Use 'interventionfeatures <command> --help' for command options.")
        sys.exit(1)

    command = sys.argv[1]

    if command == "config":
        print_config()
    elif command == "run":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run()
    elif command == "explain":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        explain()
    elif command in ["--help", "-h"]:
        print("Usage: interventionfeatures <command> [options]")
        print("")
        print("Commands:")
        print("  run      Run the CSS direction finding pipeline")
        print("  explain  Generate explanations for CSS directions")
        print("  config   Print the default configuration")
    else:
        print(f"Unknown command: {command}")
        print("Use 'interventionfeatures --help' for available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()
