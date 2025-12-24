#!/usr/bin/env python3
"""CLI for interventionfeatures - CSS direction finding for mechanistic interpretability."""

import json
import os
import pickle
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


# Compute config path - works both in development and when installed
def get_config_path():
    """Get the path to the conf directory."""
    # Try to find conf relative to this file (development mode)
    current_file = Path(__file__).resolve()
    dev_conf = current_file.parent.parent.parent.parent / "conf"
    if dev_conf.exists():
        return str(dev_conf)

    # In installed mode, conf should be at the package root
    import site
    for site_dir in site.getsitepackages():
        installed_conf = Path(site_dir) / "conf"
        if installed_conf.exists():
            return str(installed_conf)

    # Fallback to relative path
    return "../../../conf"


CONFIG_PATH = get_config_path()


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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def run(cfg: DictConfig) -> None:
    """Run the CSS direction finding pipeline."""
    setup_environment()

    print("=" * 60)
    print("InterventionFeatures - CSS Direction Finding")
    print("=" * 60)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.output_dir)  # Created automatically by Hydra

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
    db_path = str(output_dir / "searcher_db")
    puffer_data_dir = cfg.database.openpuffer_data_dir or str(output_dir / "puffer_data")

    searcher = ActivationSimSearcher(
        data_handler=data_handler,
        d_model=data_handler.d_model,
        device=device,
        scoring_type=cfg.database.scoring_type,
        db_path=db_path,
        save_activations=cfg.database.save_activations,
        openpuffer_host=cfg.database.openpuffer_host,
        openpuffer_port=cfg.database.openpuffer_port,
        openpuffer_binary_path=cfg.database.openpuffer_binary_path,
        openpuffer_data_dir=puffer_data_dir,
    )

    if not searcher.is_indexed:
        searcher.build_index_incremental(
            cfg.database.index_size,
            cfg.database.index_batch_size
        )

    # Generate visualizations
    if cfg.visualization.enabled:
        print("\nGenerating visualizations...")
        from ..utils.visualization import ActivationSimDisplay, generate_index_page

        html_dir = output_dir / "html"
        html_dir.mkdir(exist_ok=True)

        # Load CSS results
        with open(results_path, "rb") as f:
            css_results = pickle.load(f)

        vis_display = ActivationSimDisplay(searcher)

        # Generate visualization for each direction
        for idx, result in enumerate(css_results):
            direction_vec = result["s"]
            polarity = result.get("polarity", "positive")

            print(f"  Generating visualization for direction {idx+1}/{len(css_results)} ({polarity})...")

            top_results, scores_by_vec = searcher.search_with_aggregation(
                direction_vec.to(device),
                use_cosine_scoring=(cfg.database.scoring_type == "cosine"),
                top_k=cfg.visualization.num_examples,
                rerank_top_n=200,
                aggregation="mean",
            )

            vis_display.set_query(direction_vec)
            output_json = html_dir / f"{idx}_{polarity}_visualization.json"

            vis_display.display_search_results(
                results=top_results,
                scores_by_vec=scores_by_vec,
                output_file_path=str(output_json),
                polarity=polarity
            )

        # Generate index page
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        generate_index_page(
            output_path=str(output_dir),
            config_dict=config_dict,
            config=cfg,
        )

        print(f"  Visualizations saved to: {html_dir}")
        print(f"  Run Streamlit: uv run interventionfeatures viz {output_dir}")

    print("\nPipeline complete!")
    print(f"Output directory: {output_dir}")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def explain(cfg: DictConfig) -> None:
    """Generate explanations for CSS directions."""
    setup_environment()

    output_dir = Path(cfg.output_dir)
    directions_path = output_dir / "directions.pkl"
    searcher_db_path = output_dir / "searcher_db"

    # Validate prerequisites
    if not directions_path.exists():
        print(f"Error: {directions_path} not found. Run 'run' command first.")
        sys.exit(1)
    if not searcher_db_path.exists():
        print(f"Error: {searcher_db_path} not found. Run 'run' command first.")
        sys.exit(1)

    # Load directions
    with open(directions_path, "rb") as f:
        css_results = pickle.load(f)

    # Setup pipeline components
    print("\nInitializing components...")
    data_handler, model_segment, device = setup_pipeline(cfg)

    # Load search index
    from ..core.search import ActivationSimSearcher

    puffer_data_dir = cfg.database.openpuffer_data_dir or str(output_dir / "puffer_data")
    searcher = ActivationSimSearcher(
        data_handler=data_handler,
        d_model=data_handler.d_model,
        device=device,
        scoring_type=cfg.database.scoring_type,
        db_path=str(searcher_db_path),
        save_activations=cfg.database.save_activations,
        openpuffer_host=cfg.database.openpuffer_host,
        openpuffer_port=cfg.database.openpuffer_port,
        openpuffer_binary_path=cfg.database.openpuffer_binary_path,
        openpuffer_data_dir=puffer_data_dir,
    )

    # Initialize explainer
    from ..analysis.explainer import FeatureExplainer
    explainer = FeatureExplainer(
        model=model_segment,
        explainer_llm_provider=cfg.llm.explainer.provider,
        explainer_llm_model_name=cfg.llm.explainer.model_name,
        temperature=0.0
    )

    print(f"\nGenerating explanations for {len(css_results)} directions...")

    explanations = []
    for i, result in enumerate(css_results):
        direction_vec = result["s"]
        polarity = result.get("polarity", "positive")

        print(f"  Direction {i+1}/{len(css_results)} ({polarity})...")

        explanation, examples, max_act = explainer.explain_css_direction(
            css_direction=direction_vec.to(device),
            searcher=searcher,
            data_handler=data_handler,
            num_examples=10
        )

        explanations.append({
            "direction_index": i,
            "polarity": polarity,
            "explanation": explanation,
            "max_activation": max_act,
            "examples": examples[:5]
        })

    # Save explanations
    explanations_path = output_dir / "explanations.json"
    with open(explanations_path, "w") as f:
        json.dump({"explanations": explanations}, f, indent=2)

    print(f"\nExplanations saved to: {explanations_path}")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def benchmark(cfg: DictConfig) -> None:
    """Run RAVEL and/or MIB benchmarks on CSS directions."""
    setup_environment()

    print("=" * 60)
    print("InterventionFeatures - Benchmark Evaluation")
    print("=" * 60)
    print(f"\nBenchmark Configuration:\n{OmegaConf.to_yaml(cfg.benchmark)}")

    from ..config import setup_device

    output_dir = Path(cfg.output_dir)
    directions_path = output_dir / cfg.benchmark.directions_path

    if not directions_path.exists():
        print(f"Error: directions not found at {directions_path}")
        print("Run the 'run' command first to generate CSS directions.")
        sys.exit(1)

    # Load CSS directions
    with open(directions_path, "rb") as f:
        css_directions = pickle.load(f)

    print(f"Loaded {len(css_directions)} CSS directions")

    device = setup_device()
    results = {}

    # Run RAVEL if enabled
    if "ravel" in cfg.benchmark.enabled_benchmarks:
        print("\n--- Running RAVEL Benchmark ---")
        from ..benchmarks.ravel import RAVELBenchmarkRunner

        ravel_runner = RAVELBenchmarkRunner(
            css_directions=css_directions,
            model_name=cfg.model.model_name,
            layer=cfg.model.layer_cutoff,
            device=device,
            ravel_repo_path=cfg.benchmark.ravel.ravel_repo_path,
        )

        ravel_results = ravel_runner.run_all_evaluations(
            direction_indices=cfg.benchmark.direction_indices,
            entity_types=list(cfg.benchmark.ravel.entity_types)
            if cfg.benchmark.ravel.entity_types
            else None,
        )
        results["ravel"] = ravel_results

        # Print summary
        print("\nRAVEL Results:")
        for task_name, result in ravel_results.items():
            print(f"  {task_name}:")
            print(f"    CAUSE: {result.metrics.get('cause', 0):.4f}")
            print(f"    Isolation: {result.metrics.get('isolation', 0):.4f}")
            print(f"    Disentangle: {result.metrics.get('disentangle', 0):.4f}")

    # Run MIB if enabled
    if "mib" in cfg.benchmark.enabled_benchmarks:
        print("\n--- Running MIB Benchmark ---")
        from ..benchmarks.mib import MIBBenchmarkRunner

        mib_runner = MIBBenchmarkRunner(
            css_directions=css_directions,
            model_name=cfg.model.model_name,
            layer=cfg.model.layer_cutoff,
            device=device,
            hf_cache_dir=cfg.benchmark.mib.hf_cache_dir,
        )

        # Get tasks to run
        tasks_to_run = (
            list(cfg.benchmark.mib.tasks)
            if cfg.benchmark.mib.tasks
            else None
        )

        num_samples = (
            cfg.benchmark.mib.max_samples_per_task
            if cfg.benchmark.mib.max_samples_per_task > 0
            else None
        )

        mib_results = mib_runner.run_all_evaluations(
            direction_indices=cfg.benchmark.direction_indices,
            tasks=tasks_to_run,
            num_samples=num_samples,
        )
        results["mib"] = mib_results

        # Print summary
        print("\nMIB Results:")
        for task_name, result in mib_results.items():
            print(f"  {task_name}: IIA = {result.metrics.get('iia', 0):.4f}")

        # Print aggregate
        aggregate_iia = mib_runner.get_aggregate_score(mib_results)
        print(f"\n  Aggregate IIA: {aggregate_iia:.4f}")

    # Save results
    if results:
        _save_benchmark_results(results, output_dir, cfg.benchmark.output_format)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def discover(cfg: DictConfig) -> None:
    """Discover task-specific CSS directions for benchmark tasks.

    This command implements task-targeted CSS direction finding, which optimizes
    directions to maximize Interchange Intervention Accuracy (IIA) for specific
    benchmark tasks, rather than finding generic high-sensitivity features.
    """
    setup_environment()

    print("=" * 60)
    print("Task-Targeted CSS Direction Discovery")
    print("=" * 60)
    print(f"\nDiscover Configuration:\n{OmegaConf.to_yaml(cfg.discover)}")

    from transformer_lens import HookedTransformer

    from ..config import set_global_seed, setup_device
    from ..core.task_css import TaskTargetedCSSFinder

    set_global_seed(cfg.seed)
    device = setup_device()

    output_dir = Path(cfg.output_dir)

    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"Config saved to: {config_path}")

    # Load model
    print(f"\nLoading model: {cfg.model.model_name}")
    model = HookedTransformer.from_pretrained(
        cfg.model.model_name,
        device=device,
    )
    model.eval()

    # Discover directions for each task
    tasks = list(cfg.discover.tasks)
    all_results = {}

    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")

        try:
            finder = TaskTargetedCSSFinder(
                model=model,
                task_name=task_name,
                layer=cfg.model.layer_cutoff,
                device=device,
                max_seq_len=cfg.model.max_seq_len,
                hf_cache_dir=cfg.discover.hf_cache_dir,
            )

            # Find direction(s) for this task
            if cfg.discover.num_directions_per_task > 1:
                results = finder.find_multiple_directions(
                    num_directions=cfg.discover.num_directions_per_task,
                    num_iterations=cfg.discover.iterations,
                    batch_size=cfg.discover.batch_size,
                    learning_rate=cfg.discover.learning_rate,
                    target_norm=cfg.discover.target_norm,
                    orthogonality_weight=cfg.discover.orthogonality_weight,
                )
            else:
                result = finder.find_task_direction(
                    num_iterations=cfg.discover.iterations,
                    batch_size=cfg.discover.batch_size,
                    learning_rate=cfg.discover.learning_rate,
                    target_norm=cfg.discover.target_norm,
                    eval_freq=cfg.discover.eval_freq,
                )
                results = [result]

            all_results[task_name] = results

            # Print summary
            for i, r in enumerate(results):
                print(f"  Direction {i+1}: IIA = {r['iia']:.4f}")

        except Exception as e:
            print(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    save_path = output_dir / "task_directions.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Discovery Summary")
    print("=" * 60)
    for task_name, results in all_results.items():
        avg_iia = sum(r["iia"] for r in results) / len(results)
        print(f"  {task_name}: {len(results)} direction(s), avg IIA = {avg_iia:.4f}")

    print(f"\nOutput directory: {output_dir}")
    print("\nTo evaluate on benchmarks, run:")
    print("  uv run interventionfeatures benchmark benchmark.directions_path=task_directions.pkl")


def _save_benchmark_results(
    results: dict,
    output_dir: Path,
    output_format: str,
) -> None:
    """Save benchmark results to file."""
    results_path = output_dir / f"benchmark_results.{output_format}"

    if output_format == "json":
        # Convert BenchmarkResult objects to dicts
        serializable = {}
        for bench_name, bench_results in results.items():
            serializable[bench_name] = {}
            for task_name, result in bench_results.items():
                serializable[bench_name][task_name] = result.to_dict()

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    elif output_format == "pickle":
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

    else:
        # Default to JSON
        results_path = output_dir / "benchmark_results.json"
        serializable = {}
        for bench_name, bench_results in results.items():
            serializable[bench_name] = {}
            for task_name, result in bench_results.items():
                serializable[bench_name][task_name] = result.to_dict()

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


def print_config():
    """Print the default configuration."""
    from dataclasses import asdict

    from ..config import Config

    cfg = Config()
    print(OmegaConf.to_yaml(OmegaConf.create(asdict(cfg))))


def viz():
    """Launch Streamlit visualization app."""
    import subprocess

    app_path = Path(__file__).parent.parent / "utils" / "streamlit_app.py"
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--", *args])


def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: interventionfeatures <command> [options]")
        print("")
        print("Commands:")
        print("  run        Run the CSS direction finding pipeline")
        print("  discover   Discover task-targeted CSS directions for benchmarks")
        print("  explain    Generate explanations for CSS directions")
        print("  benchmark  Run RAVEL and MIB benchmark evaluations")
        print("  viz        Launch Streamlit visualization app")
        print("  config     Print the default configuration")
        print("")
        print("Use 'interventionfeatures <command> --help' for command options.")
        sys.exit(1)

    command = sys.argv[1]

    if command == "config":
        print_config()
    elif command == "run":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run()
    elif command == "discover":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        discover()
    elif command == "explain":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        explain()
    elif command == "benchmark":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        benchmark()
    elif command == "viz":
        viz()
    elif command in ["--help", "-h"]:
        print("Usage: interventionfeatures <command> [options]")
        print("")
        print("Commands:")
        print("  run        Run the CSS direction finding pipeline")
        print("  discover   Discover task-targeted CSS directions for benchmarks")
        print("  explain    Generate explanations for CSS directions")
        print("  benchmark  Run RAVEL and MIB benchmark evaluations")
        print("  viz        Launch Streamlit visualization app")
        print("  config     Print the default configuration")
    else:
        print(f"Unknown command: {command}")
        print("Use 'interventionfeatures --help' for available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()
