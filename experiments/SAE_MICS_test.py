#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script compares intervention scores of SAE-derived feature directions against
random directions for a batch of model configurations. It imports a list of
configurations, evaluates directions for each, and outputs a series of
Typst-formatted tables with statistical comparisons.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn
from sae_lens import SAE

# Add src to path for imports, assuming script is run from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the list of configurations to run
from exp_configs import config_files
from interventionfeatures.core.css import CSSDirectionFinder
from interventionfeatures.utils.config import MainConfig
from interventionfeatures.cli import main as cli_module


def setup_components(config_path: str) -> Tuple[MainConfig, CSSDirectionFinder, torch.device]:
    """Loads configuration and sets up the main pipeline components."""
    config = MainConfig.from_file(config_path)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        data_handler, model_segment, device = cli_module.setup_pipeline_components(config, progress)
    css = CSSDirectionFinder.from_config(config, data_handler, model_segment)
    return config, css, device


def get_sae_directions(config: MainConfig, num_features: int) -> torch.Tensor:
    """Loads the specified SAE and extracts its encoder weights."""
    print(f"✅ Loading SAE '{config.model_name_sae}' for layer {config.layer_cutoff}...")
    sae, _, _ = SAE.from_pretrained(
        release=config.model_name_sae,
        sae_id=f"blocks.{config.layer_cutoff}.{config.hook_type}",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Transpose W_enc to get directions of shape [n_features, d_model]
    sae_directions = sae.W_enc.T
    dirs = torch.randperm(sae_directions.shape[0])[:num_features]
    # Select num_features indices randomly 

    return sae_directions[dirs]


def generate_random_directions(num_features: int, dim: int) -> torch.Tensor:
    """Generates random directions in the activation space."""
    return torch.randn(num_features, dim)


def evaluate_directions(
    css: CSSDirectionFinder,
    directions: torch.Tensor,
    target_norm: float,
    num_samples_eval: int,
    device: torch.device,
) -> List[float]:
    """Evaluates the intervention score (J_s) for a batch of directions."""
    scores = []
    with Progress(transient=True) as progress:
        task = progress.add_task("[cyan]Evaluating directions...", total=len(directions))
        for direction_vec in directions:
            normalized_vec = direction_vec * (target_norm / torch.linalg.norm(direction_vec))
            vec_to_eval = normalized_vec.to(device).unsqueeze(0)
            score = css._evaluate_J_s(vec_to_eval, num_samples_eval)
            scores.append(score)
            progress.update(task, advance=1, description=f"[cyan]Evaluating directions... Score: {score:.4f}")
    return scores


def calculate_stats(scores: List[float]) -> Dict[str, float]:
    """Calculates mean, standard deviation, and 95% confidence interval."""
    if not scores:
        return {"mean": 0, "std": 0, "n": 0, "ci_lower": 0, "ci_upper": 0}

    scores_tensor = torch.tensor(scores)
    n = len(scores)
    mean = scores_tensor.mean().item()
    std = scores_tensor.std().item()
    sem = std / (n**0.5) if n > 0 else 0
    z_score = 1.96  # For 95% confidence
    ci_lower = mean - z_score * sem
    ci_upper = mean + z_score * sem

    return {"mean": mean, "std": std, "n": n, "ci_lower": ci_lower, "ci_upper": ci_upper}


def generate_typst_table(all_stats: List[Tuple[Dict, Dict, MainConfig]]) -> str:
    """Generates a Typst-formatted table from the statistics, including a caption."""

    typst_string = f"""table(
      columns: (auto, auto, auto, auto, auto),
      inset: 10pt,
      align: center,
      table.header(
        [],
        [Direction Type],
        [Mean Score],
        [Std. Deviation],
        [95% Confidence Interval],
      ),
    """

    def format_ci(stats: Dict) -> str:
        return f"({stats['ci_lower']:.2e}, {stats['ci_upper']:.2e})"

    for i, (sae_stats, random_stats, config) in enumerate(all_stats):
        name = config.model_name
        typst_string += f"\n table.cell(rowspan: 2)[{name}, Layer {config.layer_cutoff}],"
        typst_string += f"\n [SAE Features n={sae_stats['n']})],"
        typst_string +=  f"\n [{sae_stats['mean']:.2e}], [{sae_stats['std']:.2e}], [{format_ci(sae_stats)}]"
        typst_string += f"\n [Random Directions Features n={random_stats['n']})],"
        typst_string +=  f"\n [{random_stats['mean']:.2e}], [{random_stats['std']:.2e}], [{format_ci(random_stats)}]"
    typst_string += "\n )"
    return typst_string


def calculate_aggregate_stats(all_stats: List[Tuple[Dict, Dict, MainConfig]]) -> Dict:
    """Calculates aggregate statistics across all layers."""
    if not all_stats:
        return {
            "avg_random_score": 0.0,
            "avg_sae_score": 0.0,
            "avg_multiplicative_difference": 0.0,
            "layer_details": []
        }
    
    random_means = []
    sae_means = []
    layer_details = []
    
    for sae_stats, random_stats, config in all_stats:
        random_mean = random_stats['mean']
        sae_mean = sae_stats['mean']
        
        random_means.append(random_mean)
        sae_means.append(sae_mean)
        
        # Calculate multiplicative difference for this layer
        multiplicative_diff = sae_mean / random_mean if random_mean != 0 else 0.0
        
        layer_details.append({
            "layer": config.layer_cutoff,
            "model_name": config.model_name,
            "random_mean": random_mean,
            "sae_mean": sae_mean,
            "multiplicative_difference": multiplicative_diff
        })
    
    # Calculate averages across all layers
    avg_random_score = sum(random_means) / len(random_means)
    avg_sae_score = sum(sae_means) / len(sae_means)
    avg_multiplicative_difference = sum(s / r for (s, r) in zip(sae_means, random_means)) / len(sae_means) 
    #avg_sae_score / avg_random_score if avg_random_score != 0 else 0.0
    
    return {
        "avg_random_score": avg_random_score,
        "avg_sae_score": avg_sae_score,
        "avg_multiplicative_difference": avg_multiplicative_difference,
        "layer_details": layer_details
    }


def main():
    """Main function to run the comparison and generate the output table."""
    parser = argparse.ArgumentParser(
        description="Compare SAE feature directions to random directions for multiple configs."
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=100,
        help="Number of SAE/random features to evaluate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to use for evaluating the intervention score.",
    )
    args = parser.parse_args()

    print(f"🚀 Found {len(config_files)} configurations to process.")

    all_stats = []
    for config_path in config_files:
        print(f"\n\n{'='*80}")
        print(f"Processing Configuration: {config_path}")
        print(f"{'='*80}\n")
        
        try:
            # --- 1. Setup ---
            config, css, device = setup_components(config_path)

            # --- 2. Get Directions ---
            sae_directions = get_sae_directions(config, args.num_features)
            dim = sae_directions.shape[-1]
            random_directions = generate_random_directions(args.num_features, dim)
            print(f"✅ Loaded {len(sae_directions)} SAE directions and generated {len(random_directions)} random directions.")

            # --- 3. Evaluate Scores ---
            print("\n--- Evaluating SAE Directions ---")
            sae_scores = evaluate_directions(css, sae_directions, config.target_norm, args.num_samples, device)

            print("\n--- Evaluating Random Directions ---")
            random_scores = evaluate_directions(css, random_directions, config.target_norm, args.num_samples, device)

            # --- 4. Calculate Statistics ---
            sae_stats = calculate_stats(sae_scores)
            random_stats = calculate_stats(random_scores)

            # --- 5. Generate and Print Typst Table ---
            print("\n\n📊 Comparison Results 📊")
            all_stats.append([sae_stats, random_stats, config])
        
        except Exception as e:
            raise e

    typst_table = generate_typst_table(all_stats)#list(zip(sae_stats, random_stats, config)))
    print("Copy the following Typst code into your document:")
    print("-------------------------------------------------")
    print("Final typst_table: \n", typst_table)
    print("-------------------------------------------------")
    with open("experiments/SAE.typ", "w") as f:
        f.write(typst_table)
    
    # Generate and save JSON statistics
    aggregate_stats = calculate_aggregate_stats(all_stats)
    json_output_path = "experiments/SAE_stats.json"
    with open(json_output_path, "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    
    print(f"\n📊 Aggregate Statistics:")
    print(f"Average Random Score: {aggregate_stats['avg_random_score']:.6f}")
    print(f"Average SAE Score: {aggregate_stats['avg_sae_score']:.6f}")
    print(f"Average Multiplicative Difference: {aggregate_stats['avg_multiplicative_difference']:.6f}")
    print(f"JSON stats saved to: {json_output_path}")
    
    return typst_table


# To run, use the CLI command PYTHONPATH=$(pwd)/src/ python3 experiments/SAE.py
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    print("Running SAE")
    main()