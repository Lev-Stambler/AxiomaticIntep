"""
Modal-based SAE Benchmark Runner

Run with:
    modal run scripts/modal_benchmark.py
    modal run scripts/modal_benchmark.py --sae-type pythia-160m-32k --layer 8 --num-samples 100
    modal run scripts/modal_benchmark.py --compare --num-samples 200
"""

import modal

# Create Modal app
app = modal.App("sae-benchmarks")

# Define the image with all dependencies
# Use add_local_python_source for local code (Modal 1.0+ API)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "transformer-lens>=1.0.0",
        "sae-lens>=1.0.0",
        "eai-sparsify>=0.3.0",
        "huggingface-hub>=0.20.0",
        "bitsandbytes>=0.41.0",  # Required by scheduler.py
    )
    # Mount local source code
    .add_local_dir(
        "/home/lev/code/research/ai/AxiomaticIntep/src/interventionfeatures",
        remote_path="/root/interventionfeatures",
    )
)


@app.function(
    image=image,
    gpu="T4",  # Use T4 for cost efficiency, can upgrade to A10G or A100
    timeout=1800,  # 30 minutes
)
def run_sae_benchmark(
    sae_type: str = "pythia-160m-32k",
    layer: int = 8,
    num_samples: int = 100,
    filter_examples: bool = True,
) -> dict:
    """Run SAE benchmark on Modal GPU."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from interventionfeatures.benchmarks.sae_runner import SAEBenchmarkRunner

    print(f"\n{'='*60}")
    print(f"SAE Benchmark: {sae_type}")
    print(f"Layer: {layer}")
    print(f"Samples: {num_samples}")
    print(f"Filter: {filter_examples}")
    print(f"{'='*60}\n")

    # Initialize runner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = SAEBenchmarkRunner(
        sae_type=sae_type,
        layer=layer,
        device=device,
    )

    # Run IOI evaluation
    result = runner.run_evaluation(
        "ioi",
        num_samples=num_samples,
        filter_examples=filter_examples,
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

    return {
        "task_name": result.task_name,
        "metrics": result.metrics,
        "metadata": result.metadata,
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,  # 2 hours for fair comparison
)
def run_fair_comparison(
    num_samples: int = 100,
    filter_examples: bool = True,
    model_name: str = "EleutherAI/pythia-160m-deduped",
    sae_type: str = "pythia-160m-32k",
    layer: int = 8,
    hook_type: str = "mlp_out",  # "mlp_out" or "resid_post" or "resid_pre"
) -> dict:
    """
    Run FAIR comparison with multiple baselines:
    1. Raw activation patching (100% ceiling - direct swap)
    2. SAE reconstruction + intervention (encode-swap-decode)

    This lets us measure SAE reconstruction loss impact on intervention accuracy.
    """
    import sys
    sys.path.insert(0, "/root")

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # Determine hook name based on type
    if hook_type == "mlp_out":
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif hook_type == "resid_post":
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif hook_type == "resid_pre":
        hook_name = f"blocks.{layer}.hook_resid_pre"
    else:
        raise ValueError(f"Unknown hook_type: {hook_type}")

    print(f"Hook: {hook_name}")
    print(f"Hook type: {hook_type}")

    # Load SAE (only used for SAE-based methods)
    featurizer = None
    if hook_type == "mlp_out":
        print(f"Loading SAE: {sae_type}")
        from interventionfeatures.benchmarks.sae_runner import SAEBenchmarkRunner
        sae_runner = SAEBenchmarkRunner(sae_type=sae_type, layer=layer, device=device)
        featurizer = sae_runner.get_featurizer()
        featurizer = featurizer.to(device)
        print(f"SAE latents: {featurizer.d_sae}")
    else:
        print("Skipping SAE loading (not applicable for residual stream hooks)")

    # Load dataset
    ds = load_dataset("mib-bench/ioi", split="test")
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    # Track results for each method
    results = {
        "no_intervention": {"correct": 0, "total": 0},     # Sanity check: should be 0%
        "raw_patch_diff_only": {"correct": 0, "total": 0}, # Patch only differing tokens
        "raw_patch_diff_to_end": {"correct": 0, "total": 0}, # Patch from diff to end (100% ceiling)
        "sae_reconstruction": {"correct": 0, "total": 0},  # SAE ceiling: encode source → decode → patch
        "sae_intervention": {"correct": 0, "total": 0},
        "random_subspace_1": {"correct": 0, "total": 0},   # CSS-style with 1 random direction
        "random_subspace_10": {"correct": 0, "total": 0},  # CSS-style with 10 random directions
        "random_subspace_50": {"correct": 0, "total": 0},  # CSS-style with 50 random directions
    }
    filtered_out = 0
    debug_printed = 0  # Limit debug output

    # Create random orthonormal directions for CSS-style baseline
    d_model = model.cfg.d_model
    random_dirs_1 = torch.randn(1, d_model, device=device)
    random_dirs_1 = random_dirs_1 / random_dirs_1.norm(dim=-1, keepdim=True)

    random_dirs_10 = torch.randn(10, d_model, device=device)
    Q, _ = torch.linalg.qr(random_dirs_10.T)
    random_dirs_10 = Q.T[:10]

    random_dirs_50 = torch.randn(50, d_model, device=device)
    Q, _ = torch.linalg.qr(random_dirs_50.T)
    random_dirs_50 = Q.T[:50]

    def css_style_intervention(base_act, source_act, directions):
        """CSS-style intervention: swap projection, preserve orthogonal complement."""
        # Project onto directions
        base_proj = torch.einsum("...d,nd->...n", base_act, directions)
        source_proj = torch.einsum("...d,nd->...n", source_act, directions)

        # Compute orthogonal component of base
        base_in_subspace = torch.einsum("...n,nd->...d", base_proj, directions)
        base_orthogonal = base_act - base_in_subspace

        # Reconstruct with source's projection + base's orthogonal
        source_in_subspace = torch.einsum("...n,nd->...d", source_proj, directions)
        return source_in_subspace + base_orthogonal

    for example in tqdm(ds, desc="Fair Comparison"):
        # Base input and expected answer
        base_input = example["prompt"]
        base_choices = example["choices"]
        base_answer_idx = example["answerKey"]
        base_expected = base_choices[base_answer_idx]

        # Source input - try s2_io_flip which should change the answer
        # If s2_io_flip gives same answer as base, skip this example
        cf = example.get("s2_io_flip_counterfactual") or example["s1_io_flip_counterfactual"]
        source_input = cf["prompt"]
        source_answer_idx = cf["answerKey"]
        source_expected = cf["choices"][source_answer_idx]

        # CRITICAL: Only evaluate examples where the answer actually changes!
        if base_expected == source_expected:
            filtered_out += 1
            continue

        # Tokenize
        base_tokens = model.tokenizer(base_input, return_tensors="pt")
        base_ids = base_tokens["input_ids"].to(device)
        source_tokens = model.tokenizer(source_input, return_tensors="pt")
        source_ids = source_tokens["input_ids"].to(device)

        # Check if model gets base correct (filtering)
        if filter_examples:
            with torch.no_grad():
                base_out = model.generate(base_ids, max_new_tokens=3, do_sample=False)
            base_pred = model.tokenizer.decode(
                base_out[0, base_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            if base_expected not in base_pred:
                filtered_out += 1
                continue

        # Find position where tokens differ (giver name position for s2_io_flip)
        # This is the key intervention position
        min_len = min(base_ids.shape[1], source_ids.shape[1])
        diff_positions = []
        for i in range(min_len):
            if base_ids[0, i] != source_ids[0, i]:
                diff_positions.append(i)

        # For fair comparison, intervene at ALL differing positions
        # (the giver name might span multiple tokens)
        if not diff_positions:
            # No difference found, skip
            if debug_printed < 3:
                print(f"  WARNING: No token differences found!")
            filtered_out += 1
            continue

        # Two strategies:
        # 1. Only diff positions (conservative)
        # 2. All positions from first diff to end (aggressive)
        prompt_len = base_ids.shape[1]
        intervention_positions_minimal = diff_positions
        intervention_positions_all = list(range(diff_positions[0], prompt_len))

        if debug_printed < 3:
            diff_tokens_base = [model.tokenizer.decode([base_ids[0, p]]) for p in diff_positions]
            diff_tokens_source = [model.tokenizer.decode([source_ids[0, p]]) for p in diff_positions]
            print(f"  Diff positions: {diff_positions}")
            print(f"  Base tokens at diff: {diff_tokens_base}")
            print(f"  Source tokens at diff: {diff_tokens_source}")

        # Get activations at intervention positions
        with torch.no_grad():
            _, base_cache = model.run_with_cache(base_ids, names_filter=[hook_name])
            _, source_cache = model.run_with_cache(source_ids, names_filter=[hook_name])

        # Collect activations for both strategies
        base_acts_minimal = [base_cache[hook_name][:, pos, :] for pos in intervention_positions_minimal]
        source_acts_minimal = [source_cache[hook_name][:, pos, :] for pos in intervention_positions_minimal]
        base_acts_all = [base_cache[hook_name][:, pos, :] for pos in intervention_positions_all]
        source_acts_all = [source_cache[hook_name][:, pos, :] for pos in intervention_positions_all]

        def make_hook_multi(source_acts_list, positions, p_len):
            def hook(activation, hook=None):
                if activation.shape[1] == p_len:
                    for i, pos in enumerate(positions):
                        activation[:, pos, :] = source_acts_list[i]
                return activation
            return hook

        # Also keep single-position for backwards compatibility
        base_act = base_cache[hook_name][:, intervention_positions_minimal[0], :]
        source_act = source_cache[hook_name][:, intervention_positions_minimal[0], :]

        def make_hook(i_act, p_len, pos):
            def hook(activation, hook=None):
                if activation.shape[1] == p_len:
                    activation[:, pos, :] = i_act
                return activation
            return hook

        # === Method 0: No intervention (sanity check - should be 0% IIA) ===
        with torch.no_grad():
            output_ids = model.generate(base_ids, max_new_tokens=3, do_sample=False)
        predicted_no_int = model.tokenizer.decode(
            output_ids[0, base_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # === Debug: Print first few examples ===
        if debug_printed < 3:
            print(f"\n--- Example {debug_printed + 1} ---")
            print(f"Base prompt: {base_input}")
            print(f"Source prompt: {source_input}")
            print(f"Base expects: '{base_expected}' | Source expects: '{source_expected}'")
            print(f"No-intervention pred: '{predicted_no_int}'")
            print(f"Available keys: {list(example.keys())[:10]}")
            if "patching_positions" in example:
                print(f"Patching positions: {example['patching_positions']}")
            debug_printed += 1

        if source_expected in predicted_no_int:
            results["no_intervention"]["correct"] += 1
        results["no_intervention"]["total"] += 1

        # === Method 1a: Raw activation patching (diff positions only) ===
        with model.hooks(fwd_hooks=[(hook_name, make_hook_multi(source_acts_minimal, intervention_positions_minimal, prompt_len))]):
            output_ids = model.generate(base_ids, max_new_tokens=3, do_sample=False)
        predicted = model.tokenizer.decode(
            output_ids[0, base_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        if source_expected in predicted:
            results["raw_patch_diff_only"]["correct"] += 1
        results["raw_patch_diff_only"]["total"] += 1

        # === Method 1b: Raw activation patching (diff to end - 100% ceiling) ===
        with model.hooks(fwd_hooks=[(hook_name, make_hook_multi(source_acts_all, intervention_positions_all, prompt_len))]):
            output_ids = model.generate(base_ids, max_new_tokens=3, do_sample=False)
        predicted = model.tokenizer.decode(
            output_ids[0, base_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        if source_expected in predicted:
            results["raw_patch_diff_to_end"]["correct"] += 1
        results["raw_patch_diff_to_end"]["total"] += 1

        # === Method 2: SAE reconstruction ceiling (encode source → decode → patch) ===
        # Only run if featurizer is available (MLP hook)
        if featurizer is not None:
            with torch.no_grad():
                reconstructed_sources = []
                for src_act in source_acts_all:
                    source_latents = featurizer.encode(src_act)
                    reconstructed_sources.append(featurizer.decode(source_latents))

            with model.hooks(fwd_hooks=[(hook_name, make_hook_multi(reconstructed_sources, intervention_positions_all, prompt_len))]):
                output_ids = model.generate(base_ids, max_new_tokens=3, do_sample=False)
            predicted = model.tokenizer.decode(
                output_ids[0, base_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            if source_expected in predicted:
                results["sae_reconstruction"]["correct"] += 1
            results["sae_reconstruction"]["total"] += 1

            # === Method 3: SAE encode-swap-decode (full latent swap) ===
            with torch.no_grad():
                intervened_acts = []
                for base_a, src_a in zip(base_acts_all, source_acts_all):
                    intervened_acts.append(featurizer(base_a, src_a))

            with model.hooks(fwd_hooks=[(hook_name, make_hook_multi(intervened_acts, intervention_positions_all, prompt_len))]):
                output_ids = model.generate(base_ids, max_new_tokens=3, do_sample=False)
            predicted = model.tokenizer.decode(
                output_ids[0, base_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            if source_expected in predicted:
                results["sae_intervention"]["correct"] += 1
            results["sae_intervention"]["total"] += 1

        # === Method 4-6: Random subspace interventions (CSS-style baseline) ===
        # Uses diff-to-end for fair comparison
        for name, dirs in [
            ("random_subspace_1", random_dirs_1),
            ("random_subspace_10", random_dirs_10),
            ("random_subspace_50", random_dirs_50),
        ]:
            with torch.no_grad():
                intervened_acts = []
                for base_a, src_a in zip(base_acts_all, source_acts_all):
                    intervened_acts.append(css_style_intervention(base_a, src_a, dirs))

            with model.hooks(fwd_hooks=[(hook_name, make_hook_multi(intervened_acts, intervention_positions_all, prompt_len))]):
                output_ids = model.generate(base_ids, max_new_tokens=3, do_sample=False)
            predicted = model.tokenizer.decode(
                output_ids[0, base_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            if source_expected in predicted:
                results[name]["correct"] += 1
            results[name]["total"] += 1

    # Compute IIA for each method
    summary = {
        "model": model_name,
        "sae_type": sae_type,
        "layer": layer,
        "hook": hook_name,
        "num_samples": num_samples,
        "filtered_out": filtered_out,
        "methods": {}
    }

    for method, data in results.items():
        iia = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        summary["methods"][method] = {
            "iia": iia,
            "correct": data["correct"],
            "total": data["total"],
        }

    # Print results
    print(f"\n{'='*60}")
    print("FAIR COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"SAE: {sae_type} (layer {layer})")
    print(f"Examples evaluated: {results['raw_patch_diff_to_end']['total']}")
    print(f"Examples filtered: {filtered_out}")
    print(f"\n{'Method':<25} {'IIA':>10} {'Correct':>10}")
    print("-" * 50)
    for method, data in summary["methods"].items():
        print(f"{method:<25} {data['iia']:>10.3f} {data['correct']:>10}/{data['total']}")
    print(f"{'='*60}\n")

    return summary


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 1 hour for full comparison
)
def run_comparison(
    num_samples: int = 100,
    filter_examples: bool = True,
) -> dict:
    """Run comparison between pythia-70m and pythia-160m SAEs."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    from interventionfeatures.benchmarks.sae_runner import SAEBenchmarkRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # Test configs: (sae_type, layer)
    configs = [
        ("pythia-70m-32k", 3),   # pythia-70m with sparsify SAE
        ("pythia-160m-32k", 8),  # pythia-160m with sparsify SAE
    ]

    for sae_type, layer in configs:
        print(f"\n{'='*60}")
        print(f"Testing {sae_type} at layer {layer}")
        print(f"{'='*60}")

        try:
            runner = SAEBenchmarkRunner(
                sae_type=sae_type,
                layer=layer,
                device=device,
            )

            result = runner.run_evaluation(
                "ioi",
                num_samples=num_samples,
                filter_examples=filter_examples,
            )

            results[sae_type] = {
                "layer": layer,
                "iia": result.metrics.get("iia", 0.0),
                "examples_evaluated": result.metadata.get("examples_evaluated", 0),
                "examples_filtered": result.metadata.get("examples_filtered", 0),
            }

            print(f"IIA: {result.metrics.get('iia', 0.0):.3f}")
            print(f"Evaluated: {result.metadata.get('examples_evaluated', 0)}")
            print(f"Filtered: {result.metadata.get('examples_filtered', 0)}")

        except Exception as e:
            print(f"Error with {sae_type}: {e}")
            import traceback
            traceback.print_exc()
            results[sae_type] = {"error": str(e)}

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for sae_type, data in results.items():
        if "error" in data:
            print(f"{sae_type}: ERROR - {data['error']}")
        else:
            print(f"{sae_type} (L{data['layer']}): IIA={data['iia']:.3f} ({data['examples_evaluated']} examples)")
    print(f"{'='*60}")

    return results


@app.local_entrypoint()
def main(
    sae_type: str = "pythia-160m-32k",
    layer: int = 8,
    num_samples: int = 100,
    compare: bool = False,
    fair: bool = False,
    no_filter: bool = False,
    hook_type: str = "mlp_out",
):
    """
    Run SAE benchmarks on Modal.

    Examples:
        modal run scripts/modal_benchmark.py
        modal run scripts/modal_benchmark.py --sae-type pythia-160m-32k --layer 8
        modal run scripts/modal_benchmark.py --compare --num-samples 200
        modal run scripts/modal_benchmark.py --fair --num-samples 200
    """
    if fair:
        print(f"Running FAIR comparison (raw patching vs SAE intervention)...")
        print(f"  Hook type: {hook_type}")
        results = run_fair_comparison.remote(
            num_samples=num_samples,
            filter_examples=not no_filter,
            sae_type=sae_type,
            layer=layer,
            hook_type=hook_type,
        )
        print("\nFair Comparison Results:")
        for method, data in results["methods"].items():
            print(f"  {method}: IIA={data['iia']:.3f} ({data['correct']}/{data['total']})")
    elif compare:
        print("Running comparison between pythia-70m and pythia-160m SAEs...")
        results = run_comparison.remote(
            num_samples=num_samples,
            filter_examples=not no_filter,
        )
        print("\nFinal Results:")
        for sae_type, data in results.items():
            print(f"  {sae_type}: {data}")
    else:
        print(f"Running benchmark for {sae_type}...")
        result = run_sae_benchmark.remote(
            sae_type=sae_type,
            layer=layer,
            num_samples=num_samples,
            filter_examples=not no_filter,
        )
        print("\nFinal Result:")
        print(f"  IIA: {result['metrics'].get('iia', 'N/A')}")
        print(f"  Metadata: {result['metadata']}")
