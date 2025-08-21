#!/usr/bin/env python3
"""
FourierFeatures CLI - A professional command-line interface for mechanistic
interpretability using Contrastive Steering and Similarity (CSS).

This module provides a robust CLI for executing the various stages of the
FourierFeatures pipeline, including finding feature directions, building similarity
indexes, generating explanations, and validating results.
"""
# ruff: noqa: B904
import sys
import os
# Fix sqlite3 module for ChromaDB compatibility before any other imports.
# This is a necessary workaround for some environments.
from ..utils.visualization import ActivationSimDisplay, generate_index_page
from ..utils.config import MainConfig, generate_param_hash, setup_device, set_global_seed
from ..core.search import ActivationSimSearcher
from ..core.model import IntervenableTransformerSegment
from ..core.data_handler import TransformerDataHandler
from ..core.css import CSSDirectionFinder
from ..analysis.explainer import FeatureExplainer
from rich.traceback import install
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.console import Console
import typer
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pickle
import logging
import json
from copy import deepcopy
import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


# Local application imports (assuming a package structure)


# --- Pre-computation and Setup ---

# Install rich traceback handler for beautiful and informative error reports.
install(show_locals=True)

# Initialize console and Typer app for the CLI.
console = Console()
app = typer.Typer(
    name="interventionfeatures",
    help="A toolkit for mechanistic interpretability using intervention features and CSS.",
    add_completion=False,
)

# --- Constants ---
DIRECTIONS_FILENAME = "directions.pkl"
SEARCHER_DB_FILENAME_PKL = "searcher_db.pkl"
SEARCHER_DB_DIRNAME = "searcher_db"
EXPLANATIONS_FILENAME = "explanations.json"
METADATA_FILENAME = "metadata.json"
DEFAULT_CONFIG_FILENAME = "default_config.toml"


# --- Helper Functions: Environment & Logging ---


def load_environment_variables() -> None:
    """Load environment variables from a .env file with a graceful fallback."""
    try:
        from dotenv import load_dotenv

        if load_dotenv(dotenv_path=".env"):
            console.print("[dim]✓ Loaded environment variables from .env[/dim]")
    except ImportError:
        console.print(
            "[yellow]Warning: python-dotenv not found. "
            "Skipping .env file. Please ensure environment variables are set manually if needed.[/yellow]"
        )
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load .env file: {e}[/yellow]")


def setup_logging(verbose: bool = False, name: str = __name__) -> logging.Logger:
    """Configure logging with rich for pretty and informative output.

    Args:
        verbose: If True, set logging level to DEBUG. Otherwise, INFO.
        name: The name of the logger to configure.

    Returns:
        The configured logger instance.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = RichHandler(console=console, rich_tracebacks=True, show_path=False)
    handler.setLevel(level)

    # Optional: If you want to format the message in the handler
    # formatter = logging.Formatter('%(message)s')
    # handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # Prevent logs from propagating to the root logger

    return logger


# --- Helper Functions: Configuration & File I/O ---


def load_config(config_file: Optional[str], logger: logging.Logger) -> MainConfig:
    """Load configuration from a file or use defaults with a standard fallback logic.

    Args:
        config_file: Path to the configuration file.
        logger: The logger instance for status updates.

    Returns:
        The loaded or default MainConfig object.

    Raises:
        typer.Exit: If the configuration cannot be loaded.
    """
    try:
        if config_file:
            config = MainConfig.from_file(config_file)
            logger.info(f"Loaded configuration from: {config_file}")
            return config

        # Attempt to load the default config from the project root
        default_config_path = (
            Path(__file__).parent.parent.parent.parent / DEFAULT_CONFIG_FILENAME
        )
        if default_config_path.exists():
            config = MainConfig.from_file(default_config_path)
            logger.info(f"Using default configuration from: {default_config_path}")
            return config

        # Fallback to built-in defaults if no file is found
        logger.warning(
            f"No configuration file specified and {DEFAULT_CONFIG_FILENAME} not found. "
            "Using built-in defaults."
        )
        return MainConfig()
    except Exception as e:
        logger.error(f"Failed to load or create configuration: {e}")
        raise typer.Exit(1)


def generate_output_dirname(config: MainConfig, stage: str) -> str:
    """Generate a deterministic output directory name based on a config hash.

    Args:
        config: The main configuration object.
        stage: The pipeline stage (e.g., 'directions', 'search').

    Returns:
        The generated directory name string.
    """
    if stage == "directions":
        params_dict = config.get_data_params_dict()
    elif stage == "search":
        params_dict = config.get_db_params_dict()
    else:
        params_dict = config.get_data_params_dict()

    config_hash = generate_param_hash(params_dict)
    if stage in ["explanations", "validation"]:
        stage = "results"
    return f"{stage}_{config_hash}"


def setup_output_directory(
    config: MainConfig, stage: str, output_dir: Optional[str] = None
) -> Path:
    """Create and return the path to the output directory for a given stage.

    If output_dir is not provided, a hash-based directory name is generated
    inside the 'outputs' directory.

    Args:
        config: The main configuration object.
        stage: The pipeline stage name.
        output_dir: An optional custom output directory path.

    Returns:
        The Path object for the output directory.
    """
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path("outputs") / generate_output_dirname(config, stage)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_pickle(data: Any, filepath: Path, logger: logging.Logger):
    """Save data to a file using pickle."""
    try:
        with filepath.open("wb") as f:
            pickle.dump(data, f)
        logger.info(f"Successfully saved results to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pickle file to {filepath}: {e}")
        raise typer.Exit(1)


def load_pickle(filepath: Path, logger: logging.Logger) -> Any:
    """Load data from a pickle file."""
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise typer.Exit(1)
    try:
        with filepath.open("rb") as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded results from: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load pickle file from {filepath}: {e}")
        raise typer.Exit(1)


def save_json(data: Dict, filepath: Path, logger: logging.Logger):
    """Save a dictionary to a JSON file."""
    try:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JSON data to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON file to {filepath}: {e}")
        raise typer.Exit(1)


def load_json(filepath: Path, logger: logging.Logger) -> Dict:
    """Load data from a JSON file."""
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise typer.Exit(1)
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON data from: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file from {filepath}: {e}")
        raise typer.Exit(1)


# --- Helper Functions: Artifact Discovery ---


def find_artifact_path(config: MainConfig, stage: str, filename: str) -> Optional[Path]:
    """Find an artifact file in its expected hash-based directory.

    Args:
        config: The configuration used to generate the hash.
        stage: The pipeline stage (e.g., 'directions').
        filename: The name of the file to find.

    Returns:
        The Path to the file if found, otherwise None.
    """
    expected_dir = Path("outputs") / generate_output_dirname(config, stage)
    if expected_dir.is_dir():
        artifact_file = expected_dir / filename
        if artifact_file.is_file():
            return artifact_file
    return None


def find_search_dir(config: MainConfig) -> Path:
    """Find or create the search directory based on a config hash."""
    search_dir_name = generate_output_dirname(config, "search")
    search_path = Path("outputs") / search_dir_name
    search_path.mkdir(parents=True, exist_ok=True)
    return search_path


def resolve_artifact_path(
    cli_path: Optional[str],
    config: MainConfig,
    stage: str,
    filename: str,
    logger: logging.Logger,
) -> Path:
    """Resolve the path to an artifact, using auto-discovery as a fallback.

    Args:
        cli_path: The path provided via the CLI option.
        config: The main configuration object.
        stage: The pipeline stage for auto-discovery.
        filename: The filename for auto-discovery.
        logger: The logger for status messages.

    Returns:
        The resolved Path object for the artifact.

    Raises:
        typer.Exit: If the path is not provided and auto-discovery fails.
    """
    if cli_path:
        return Path(cli_path)

    logger.info(f"Attempting to auto-discover {filename}...")
    artifact_path = find_artifact_path(config, stage, filename)
    if artifact_path:
        logger.info(f"Auto-discovered artifact: {artifact_path}")
        return artifact_path

    expected_dir = generate_output_dirname(config, stage)
    logger.error(
        f"Could not find required file '{filename}' in expected directory: {expected_dir}\n"
        f"Please provide the path manually or ensure the '{stage}' stage was run with the same configuration."
    )
    raise typer.Exit(1)


# --- Helper Functions: Core Pipeline Components ---


def setup_pipeline_components(
    config: MainConfig, progress: Progress
) -> Tuple[TransformerDataHandler, IntervenableTransformerSegment, Any]:
    """Initialize and set up the main components required for the pipeline.

    This includes the device, dataset, data handler, and model segment.

    Args:
        config: The main configuration object.
        progress: The rich Progress instance for status updates.

    Returns:
        A tuple containing the initialized (data_handler, model_segment, device).

    Raises:
        typer.Exit: If any component fails to initialize.
    """
    task = progress.add_task("Setting up pipeline components...", total=5)

    def update_progress(description: str, advance: int = 1):
        progress.update(task, description=description, advance=advance)

    try:
        # 1. Set global random seed
        update_progress("Setting global random seed...")
        set_global_seed(config.seed)
        update_progress(f"Global seed set to {config.seed} ✓")

        # 2. Setup device
        update_progress("Detecting and setting up device...")
        device = setup_device()
        update_progress(f"Using device: {device} ✓")

        # 3. Load dataset
        from datasets import load_dataset

        if not os.environ.get("HUGGINGFACE_API_KEY"):
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            data_files=config.dataset_data_files if config.dataset_data_files else None,
            split=config.dataset_split,
            streaming=config.dataset_streaming,
            token=os.environ.get("HUGGINGFACE_API_KEY")
        )
        update_progress(f"Dataset '{config.dataset_name}' loaded ✓")

        # 4. Setup data handler
        data_handler = TransformerDataHandler(
            dataset=dataset,
            dataset_text_column=config.dataset_text_column,
            model_name=config.model_name,
            layer_cutoff=config.layer_cutoff,
            device=device,
            max_seq_len=config.max_seq_len,
            hook_type=config.hook_type,
            seed=config.seed,
            force_dataset_download=config.force_dataset_download,
        )
        update_progress("Data handler ready ✓")

        # 5. Setup model segment
        target_layers = config.target_layers if config.target_layers is not None else [config.layer_cutoff + 1]
        model_segment = IntervenableTransformerSegment(
            base_model=data_handler.model,
            data_handler=data_handler,
            layer_cutoff=config.layer_cutoff,
            target_layers=target_layers,
            include_final_logits=config.use_output_logits,
            target_token_offset=config.target_token_offset,
        )
        update_progress("Model segment ready ✓")

        progress.remove_task(task)
        return data_handler, model_segment, device

    except Exception as e:
        progress.update(task, description="❌ Pipeline setup failed")
        console.log(f"Error during pipeline component setup: {e}")
        if "out of memory" in str(e).lower():
            console.print(
                "[yellow]Hint: An out-of-memory error occurred. Try reducing batch sizes or using a smaller model.[/yellow]"
            )
        raise e
        raise typer.Exit(1)


def create_and_build_searcher(
    config: MainConfig,
    data_handler: TransformerDataHandler,
    device: Any,
    progress: Progress,
) -> ActivationSimSearcher:
    """Initialize the similarity searcher and build its index.

    Args:
        config: The main configuration object.
        data_handler: The data handler instance.
        device: The compute device.
        progress: The rich Progress instance for status updates.

    Returns:
        An initialized and built ActivationSimSearcher instance.
    """
    task = progress.add_task("Initializing similarity search...", total=1)
    try:
        searcher_path = find_search_dir(config)
        searcher = ActivationSimSearcher(
            data_handler=data_handler,
            d_model=data_handler.d_model,
            device=device,
            scoring_type=config.scoring_type,
            chroma_db_path=str(searcher_path),
            save_activations=config.save_activations,
        )
        progress.update(task, description="Building similarity search index...")
        if not searcher.is_indexed:
            console.log("Search index not found or incomplete. Rebuilding index...")
            searcher.build_index_incremental(config.index_size, config.index_batch_size)
        else:
            console.log("Loaded existing search index.")

        progress.update(
            task, description="Similarity search index built ✓", completed=1
        )
        return searcher
    except Exception as e:
        progress.update(task, description="❌ Similarity search setup failed")
        console.log(f"Error creating similarity searcher: {e}")
        raise typer.Exit(1)


def generate_visualizations(
    css_results: List[Dict],
    searcher: ActivationSimSearcher,
    output_path: Path,
    logger: logging.Logger,
    progress: Progress,
    explanations: Optional[List[str]] = None,
) -> List[str]:
    """Generate and save HTML visualizations for CSS directions.

    Args:
        css_results: A list of CSS direction result dictionaries.
        searcher: The similarity searcher instance.
        output_path: The directory to save the visualization files.
        logger: The logger instance.
        progress: The rich Progress instance for status updates.
        explanations: Optional list of explanations corresponding to each direction.

    Returns:
        A list of file paths for the generated visualizations.
    """
    # Create html subdirectory for consolidated visualization files
    html_dir = output_path / "html"
    html_dir.mkdir(exist_ok=True)

    # Extract faithfulness data for optimal threshold
    from ..utils.visualization import extract_faithfulness_data
    faithfulness_data = extract_faithfulness_data(str(output_path))
    direction_thresholds = faithfulness_data.get("direction_thresholds", {})

    task = progress.add_task("Generating visualizations...", total=len(css_results))
    displayer = ActivationSimDisplay(searcher)
    generated_files = []

    for i, css_result in enumerate(css_results):
        progress.update(
            task,
            description=f"Generating visualization for direction {i+1}/{len(css_results)}...",
        )
        try:
            direction_vectors = css_result.get("s")
            if direction_vectors is None:
                logger.warning(
                    f"Could not extract direction vectors from CSS result {i}. Skipping."
                )
                continue

            displayer.set_query(direction_vectors)
            search_results, scores = searcher.search_with_aggregation(
                direction_vectors, top_k=20, rerank_top_n=1_000,
                use_cosine_scoring=True
            )

            # Get polarity and feature information from CSS result
            polarity = css_result.get("polarity", "positive")
            feature_id = css_result.get("feature_id", f"feature_{i}")
            base_feature_id = css_result.get("base_feature_id", i//2)  # Fallback calculation

            # Use direction-based naming to match what generate_index_page expects
            viz_file = html_dir / f"direction_{i}_visualization.html"
            explanation = (
                explanations[i] if explanations and i < len(explanations) else None
            )

            # Use direction-specific threshold if available, otherwise default to 0.0
            direction_threshold = direction_thresholds.get(i, 0.0)

            displayer.display_search_results(
                results=search_results,
                scores_by_vec=scores,
                output_file_path=str(viz_file),
                explanation=explanation,
                optimal_threshold=direction_threshold,
                polarity=polarity
            )
            generated_files.append(str(viz_file.relative_to(output_path)))
            logger.debug(f"Saved visualization for direction {i} to {viz_file}")

        except Exception as e:
            logger.warning(f"Failed to generate visualization for direction {i}: {e}")
        finally:
            progress.advance(task)

    progress.update(
        task, description=f"Generated {len(generated_files)} visualizations ✓"
    )
    return generated_files


@app.command()
def find_directions(
    config_file: Optional[str] = typer.Option(
        default=None, help="Path to configuration file."
    ),
    output_dir: Optional[str] = typer.Option(
        default=None, help="Output directory (auto-generated if not specified)."
    ),
    verbose: bool = typer.Option(default=False, help="Enable verbose logging."),
) -> None:
    """Find CSS directions using PGA and save the results."""
    logger = setup_logging(verbose, name="interventionfeatures")
    config = load_config(config_file, logger)
    output_path = setup_output_directory(config, "directions", output_dir)
    console.rule("[bold cyan]Finding CSS Directions[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        data_handler, model_segment, _ = setup_pipeline_components(config, progress)

        task = progress.add_task("Finding optimal CSS directions...", total=1)
        try:
            css_finder = CSSDirectionFinder.from_config(
                config, data_handler, model_segment
            )
            css_results = css_finder.find_optimal_s_directions()
            progress.update(
                task,
                description=f"Found {len(css_results)} CSS directions ✓",
                completed=1,
            )

            # Save results and metadata
            directions_file = output_path / DIRECTIONS_FILENAME
            save_pickle(css_results, directions_file, logger)

            metadata = {
                "config": config.get_dict(),
                "timestamp": datetime.now().isoformat(),
            }
            save_json(metadata, output_path / METADATA_FILENAME, logger)

        except Exception as e:
            progress.update(task, description="❌ Direction finding failed")
            logger.error(f"Failed to find CSS directions: {e}")
            raise e
            raise typer.Exit(1)

    console.rule("[bold green]Direction Finding Complete[/bold green]")
    console.print(f"Results saved in: {output_path.resolve()}")


@app.command()
def explain_display(
    css_results_path: Optional[str] = typer.Argument(
        default=None, help="Path to CSS results file (auto-discovered if not provided)."
    ),
    searcher_path: Optional[str] = typer.Argument(
        default=None, help="Path to searcher database directory (auto-discovered)."
    ),
    config_file: Optional[str] = typer.Option(
        default=None, help="Path to configuration file."
    ),
    output_dir: Optional[str] = typer.Option(
        default=None, help="Output directory (auto-generated if not specified)."
    ),
    num_examples: int = typer.Option(
        20, help="Number of examples to show in explanations."
    ),
    model_name: Optional[str] = typer.Option(default=None, help="OpenAI model to use"),
    verbose: bool = typer.Option(default=False, help="Enable verbose logging."),
) -> None:
    """Generate explanations and visualizations for pre-computed CSS directions."""
    logger = setup_logging(verbose, name="interventionfeatures")
    load_environment_variables()
    config = load_config(config_file, logger)

    output_path = setup_output_directory(config, "explanations", output_dir)
    console.rule("[bold cyan]Generating Explanations and Visualizations[/bold cyan]")

    # Resolve input paths
    resolved_css_path = resolve_artifact_path(
        css_results_path, config, "directions", DIRECTIONS_FILENAME, logger
    )
    search_dir = Path(searcher_path) if searcher_path else find_search_dir(config)
    if not search_dir.exists():
        logger.error(
            f"Searcher directory not found at {search_dir}. Please run the 'search-activations' stage first."
        )
        raise typer.Exit(1)
    if model_name is not None:
        config.explainer_llm_model_name = model_name

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load inputs
        css_results = load_pickle(resolved_css_path, logger)
        data_handler, model_segment, device = setup_pipeline_components(config, progress)

        # Load or create searcher
        searcher = ActivationSimSearcher(
            data_handler=data_handler,
            d_model=data_handler.d_model,
            device=device,
            scoring_type=config.scoring_type,
            chroma_db_path=str(search_dir),
            save_activations=config.save_activations,
        )
        if not searcher.is_indexed:
            logger.info("Search index not found or incomplete. Rebuilding index...")
            searcher.build_index_incremental(config.index_size, config.index_batch_size)
        else:
            logger.info("Loaded existing search index.")

        # Generate explanations
        task = progress.add_task("Generating explanations...", total=1)
        explainer = FeatureExplainer(
            model=model_segment,
            explainer_llm_provider=config.explainer_llm_provider,
            explainer_llm_model_name=config.explainer_llm_model_name,
        )

        # Generate explanations for each feature (each feature now gets its own explanation)
        explanation_results_old = explainer.explain_css_directions(
            css_directions=css_results,
            searcher=searcher,
            data_handler=data_handler,
            num_examples=config.num_similar_to_find
        )
        explanations = [res[0] for res in explanation_results_old]
        progress.update(
            task,
            description=f"Generated {len(explanations)} feature explanations ✓",
            completed=1,
        )

        # Save explanations to JSON with polarity information
        explanations_file = output_path / EXPLANATIONS_FILENAME
        explanation_data = []
        for i, (exp, css_result) in enumerate(zip(explanation_results_old, css_results)):
            explanation_data.append({
                "direction_index": i,
                "explanation": exp[0],
                "examples": exp[1],
                "max_activation": float(exp[2]),
                "polarity": css_result.get("polarity", "unknown"),
                "base_feature_id": css_result.get("base_feature_id"),
                "feature_id": css_result.get("feature_id", str(i)),
            })

        save_json(
            {
                "explanations": explanation_data,
                "params": {"num_examples": config.num_similar_to_find},
            },
            explanations_file,
            logger,
        )

        # Generate visualizations
        visualization_files = generate_visualizations(
            css_results, searcher, output_path, logger, progress, explanations
        )

        # Generate index page
        generate_index_page(
            output_path=str(output_path),
            config_dict=config.get_dict(),
            metadata={"timestamp": datetime.now().isoformat()},
            config=config,
        )

    console.rule("[bold green]Explanation and Display Complete[/bold green]")
    console.print(f"Results saved in: {output_path.resolve()}")


@app.command()
def validate(
    css_results_path: Optional[str] = typer.Option(
        default=None, help="Path to CSS results file (auto-discovered)."
    ),
    explanations_path: Optional[str] = typer.Option(
        default=None, help="Path to explanations file (auto-discovered)."
    ),
    config_file: Optional[str] = typer.Option(
        default=None, help="Path to configuration file."
    ),
    output_dir: Optional[str] = typer.Option(
        default=None, help="Output directory (auto-generated)."
    ),
    total_trials: int = typer.Option(
        5, help="Total number of faithfulness test trials to run and aggregate."
    ),
    sort_by_faithfulness: str = typer.Option(
        "aggregated_accuracy", help="Sort directions by faithfulness metric: aggregated_accuracy, trial_mean_accuracy, css_score."
    ),
    model_name: Optional[str] = typer.Option(default=None, help="OpenAI model to use"),
    compression_level: str = typer.Option(
        "standard", help="Compression level for metadata files: minimal, standard, full."
    ),
    verbose: bool = typer.Option(default=False, help="Enable verbose logging."),
) -> None:
    """Validate explanations using multi-trial faithfulness testing. Each feature is validated independently."""
    logger = setup_logging(verbose, name="interventionfeatures")
    load_environment_variables()
    config = load_config(config_file, logger)
    output_path = setup_output_directory(config, "validation", output_dir)
    console.rule("[bold cyan]Validating Explanations[/bold cyan]")

    # Resolve input paths
    resolved_css_path = resolve_artifact_path(
        css_results_path, config, "directions", DIRECTIONS_FILENAME, logger
    )
    resolved_exp_path = resolve_artifact_path(
        explanations_path, config, "explanations", EXPLANATIONS_FILENAME, logger
    )
    if model_name is not None:
        config.faithfulness_llm_model_name = model_name

    # All features are now validated independently (no special positive/negative logic needed)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load inputs
        css_results = load_pickle(resolved_css_path, logger)
        explanations_data = load_json(resolved_exp_path, logger)
        explanations = explanations_data.get("explanations", [])

        # Verify explanations are available
        if not explanations:
            logger.error("No 'explanations' key found in the provided explanations file.")
            raise typer.Exit(1)

        # Setup components
        data_handler, _, _ = setup_pipeline_components(config, progress)

        # Use single explanation format (positive/negative functionality removed)
        processed_explanations = explanations
        validation_type = "single"

        # Extract max_activations from explanations
        max_acts = [e.get("max_activation", 0) for e in processed_explanations]
        if max_acts and max_acts[0] == 0:
            logger.error(
                "Max activation not found in processed explanations. "
                f"Current explanation format has keys: {list(processed_explanations[0].keys())}\n"
                "Expected 'max_activation' field for explanation format."
            )
            raise typer.Exit(1)

        # Run validation
        from ..analysis.faithfulness import (
            run_multi_trial_faithfulness_testing,
            run_aggregated_faithfulness_visualization,
        )

        # Run single validation
        task = progress.add_task(f"Running {total_trials} validation trials...", total=total_trials + 1)

        console.print(f"[cyan]Running {total_trials} individual faithfulness trials for aggregation...[/cyan]")
        multi_run_results = []

        # Run individual trial (1 trial per "run")
        run_result = run_multi_trial_faithfulness_testing(
            css_results=css_results,
            explanations=processed_explanations,
            data_handler=data_handler,
            config=config,
            total_trials=total_trials,  # 1 trial per run
            output_dir=None,  # No individual HTML output
            progress=None,    # No nested progress bars
            use_cosine_sim=True
        )
        multi_run_results.append(run_result)

        # Aggregate all trials
        progress.update(
            task,
            description=f"Aggregating {total_trials} trials...",
            completed=total_trials,
        )

        console.print(f"[green]Aggregating results from {total_trials} trials...[/green]")
        aggregated_results = run_aggregated_faithfulness_visualization(
            multi_run_results=multi_run_results,
            output_dir=output_path,
            sort_by=sort_by_faithfulness,
        )

        # Set validation_results to the aggregated results
        validation_results = {
            "aggregated_results": aggregated_results,
            "total_trials": total_trials,
            "validation_type": validation_type,
        }

        progress.update(
            task,
            description=f"Completed {total_trials} trials and aggregation ✓",
            completed=total_trials + 1,
        )

        # Save results with optional compression
        results_data = {"validation_results": validation_results}
        results_file = output_path / "validation_results.json"

        save_json(results_data, results_file, logger)

        save_json(
            {"config": config.get_dict(), "timestamp": datetime.now().isoformat()},
            output_path / METADATA_FILENAME,
            logger,
        )

        # Generate index page for validation results
        from ..utils.visualization import find_related_direction_files

        html_dir = output_path / "html"
        html_dir.mkdir(exist_ok=True)

        direction_files = find_related_direction_files(str(output_path))
        html_files = [f"html/{f.name}" for f in html_dir.glob("*.html")]

        if html_files or direction_files:
            generate_index_page(
                output_path=str(output_path),
                config_dict=config.get_dict(),
                metadata={"timestamp": datetime.now().isoformat()},
                config=config,
            )

    console.rule("[bold green]Validation Complete[/bold green]")
    console.print(f"Results saved in: {output_path.resolve()}")

@app.command()
def generate_vis(
    config_file: Optional[str] = typer.Option(
        default=None, help="Path to configuration file."
    ),
):
    logger = setup_logging(False, name="interventionfeatures")
    config = load_config(config_file, logger)
    output_path = setup_output_directory(config, "validation", None)
    # Discover visualization files in the output directory
    from ..utils.visualization import find_related_direction_files

    html_dir = output_path / "html"
    html_dir.mkdir(exist_ok=True)  # Ensure html directory exists

    generate_index_page(
        output_path=str(output_path),
        config_dict=config.get_dict(),
        metadata={"timestamp": datetime.now().isoformat()},
        config=config,
    )


@app.command()
def sae_faithfulness(
    config_file: Optional[str] = typer.Option(
        None,
        "--config-file",
        "-c",
        help=f"Path to config file (default: {DEFAULT_CONFIG_FILENAME} in project root)."
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom output directory for results."
    ),
    num_features: int = typer.Option(
        20,
        "--num-features",
        "-n",
        help="Number of SAE features to test for faithfulness."
    ),
    trials: int = typer.Option(
        5,
        "--trials",
        "-t",
        help="Number of trials to run per feature."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
) -> None:
    """
    Run faithfulness testing on Sparse Autoencoder (SAE) features.

    This command extracts feature directions from a specified SAE, generates
    explanations for them, and tests the faithfulness of those explanations
    using automated test data generation.

    The config file must specify 'model_name_sae' for the SAE to use.
    You must also set API keys for explanation generation (OPENAI_API_KEY or ANTHROPIC_API_KEY).
    """
    load_environment_variables()
    logger = setup_logging(verbose, name="sae_faithfulness")

    try:
        # Load configuration
        config = load_config(config_file, logger)

        # Validate SAE configuration
        if not hasattr(config, 'model_name_sae') or not config.model_name_sae:
            console.print("[red]Error: config.model_name_sae must be specified for SAE faithfulness testing[/red]")
            console.print("Please add 'model_name_sae = \"your-sae-model\"' to your config file.")
            raise typer.Exit(1)

        # Check API keys
        api_key_set = False
        if config.faithfulness_llm_provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            api_key_set = True
        elif config.faithfulness_llm_provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            api_key_set = True

        if not api_key_set:
            console.print(
                f"[red]Error: {config.faithfulness_llm_provider.upper()}_API_KEY environment variable not set[/red]")
            console.print("Please set your API key for explanation generation.")
            raise typer.Exit(1)

        # Setup output directory
        output_path = setup_output_directory(config, "sae_faithfulness", output_dir)
        console.print(f"[blue]Output directory: {output_path}[/blue]")

        # Setup pipeline components
        console.print("[blue]Setting up pipeline components...[/blue]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            data_handler, model_segment, device = setup_pipeline_components(config, progress)

        console.print(f"[green]✓ Pipeline setup complete[/green]")
        console.print(f"[blue]Model: {config.model_name}[/blue]")
        console.print(f"[blue]SAE: {config.model_name_sae}[/blue]")
        console.print(f"[blue]Layer: {config.layer_cutoff}[/blue]")
        console.print(f"[blue]Features to test: {num_features}[/blue]")
        console.print(f"[blue]Trials per feature: {trials}[/blue]")

        # Run SAE faithfulness testing
        console.print("\n[blue]Running SAE faithfulness testing...[/blue]")

        from ..analysis.faithfulness import run_sae_faithfulness_testing
        config_sae = deepcopy(config)
        config_sae.scoring_type = 'dot'  # We need the scoring type to be dot product here
        search_dir = find_search_dir(config_sae)
        sae_searcher = ActivationSimSearcher(
            data_handler=data_handler,
            d_model=data_handler.d_model,
            device=device,
            scoring_type='dot',
            chroma_db_path=str(search_dir),
            save_activations=config.save_activations,
        )
        if not sae_searcher.is_indexed:
            console.log("Search index not found or incomplete. Rebuilding index...")
            sae_searcher.build_index_incremental(config.index_size, config.index_batch_size)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            results = run_sae_faithfulness_testing(
                config=config,
                data_handler=data_handler,
                model=model_segment,
                num_features=num_features,
                total_trials=trials,
                output_dir=output_path,
                progress=progress,
                searcher=sae_searcher
            )

        # Display results summary
        overall_summary = results["overall_summary"]
        console.print("\n[green]🎉 SAE Faithfulness Testing Results[/green]")
        console.print(f"[blue]Directions tested: {overall_summary['num_directions']}[/blue]")
        console.print(
            f"[blue]Total trials: {overall_summary['total_successful_trials']}/{overall_summary['total_trials']}[/blue]")
        console.print(f"[blue]Overall mean accuracy: {overall_summary['overall_mean_accuracy']:.1%}[/blue]")
        console.print(f"[blue]Standard deviation: {overall_summary['overall_std_accuracy']:.1%}[/blue]")

        if overall_summary.get('overall_accuracy_95_ci'):
            ci_lower, ci_upper = overall_summary['overall_accuracy_95_ci']
            console.print(f"[blue]95% Confidence Interval: [{ci_lower:.1%}, {ci_upper:.1%}][/blue]")

        # Save results
        results_file = output_path / "sae_faithfulness_results.json"
        with open(results_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = results.copy()
            json_results.pop('all_trials', None)  # Remove detailed trial data for size
            json.dump(json_results, f, indent=2, default=str)

        console.print(f"\n[green]✓ Results saved to: {results_file}[/green]")

        # Show HTML output locations
        html_dir = output_path / "html"
        if html_dir.exists():
            html_files = list(html_dir.glob("*.html"))
            if html_files:
                console.print(f"\n[green]📊 HTML visualizations saved to: {html_dir}[/green]")
                for html_file in html_files:
                    console.print(f"  - {html_file.name}")

        console.print(f"\n[green]🎉 SAE faithfulness testing completed successfully![/green]")

    except Exception as e:
        raise e

# --- Main Execution Guard ---
def main() -> None:
    """Main entry point for the CLI application."""
    try:
        app()
    except typer.Exit:
        # Typer handles its own exit messages, so we just pass
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        # Fallback for unexpected errors not caught elsewhere
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        console.print_exception(show_locals=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
