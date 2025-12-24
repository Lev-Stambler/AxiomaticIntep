#!/usr/bin/env python3
"""
SAE Benchmark Comparison Script

Runs SAE benchmarks on pythia-160m with EleutherAI's official SAEs.
Compares against raw activation replacement as a baseline.
"""

import argparse
import torch
from tqdm import tqdm

from interventionfeatures.benchmarks.sae_runner import SAEBenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run SAE benchmarks")
    parser.add_argument(
        "--sae-type",
        type=str,
        default="pythia-160m-32k",
        choices=["pythia-70m-32k", "pythia-160m-32k", "pythia-70m-res", "pythia-70m-mlp"],
        help="SAE configuration to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Layer to evaluate (default: 8 for pythia-160m, use 2-4 for pythia-70m)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter examples (evaluate all, not just correctly-predicted)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SAE Benchmark: {args.sae_type}")
    print(f"Layer: {args.layer}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Filter: {not args.no_filter}")
    print(f"{'='*60}\n")

    # Initialize runner
    runner = SAEBenchmarkRunner(
        sae_type=args.sae_type,
        layer=args.layer,
        device=args.device,
    )

    # Run IOI evaluation
    result = runner.run_evaluation(
        "ioi",
        num_samples=args.num_samples,
        filter_examples=not args.no_filter,
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Task: {result.task_name}")
    print(f"IIA: {result.metrics.get('iia', 'N/A'):.3f}")
    print(f"\nMetadata:")
    for k, v in result.metadata.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    main()
