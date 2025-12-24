# Benchmark Evaluations

This document provides detailed information about the benchmark evaluations available in interventionfeatures for evaluating CSS directions.

## Overview

The benchmarks module provides standardized evaluation of CSS directions using two major mechanistic interpretability benchmarks:

1. **RAVEL** - Resolving Attribute-Value Entanglements in Language models
2. **MIB** - Mechanistic Interpretability Benchmark (Causal Variable Track)

Both benchmarks use **interchange interventions** (also called activation patching) to evaluate whether learned features can isolate and manipulate specific causal variables in neural network representations.

---

## Task-Targeted CSS Discovery

### The Key Insight

The standard CSS algorithm finds directions that maximize **generic sensitivity** - how much an intervention changes model outputs (measured by KL divergence). This produces many features that may or may not be useful for specific tasks.

**Task-targeted CSS** takes a different approach: it optimizes directions to maximize **task-specific IIA** (Interchange Intervention Accuracy). Instead of finding features blindly and then evaluating them, we directly optimize for benchmark performance.

```
Standard CSS: Find many features → Evaluate on benchmarks
Task-targeted CSS: Load benchmark task → Find 1-5 features that maximize task performance
```

### How It Works

1. **Load task dataset** (e.g., IOI with base/source input pairs)
2. **Define task-specific objective**: Does intervention cause correct output?
3. **PGA optimization**: Gradient ascent on IIA, not KL divergence
4. **Output**: Task-specific direction that maximizes benchmark score

### Usage

```bash
# Discover direction for IOI task
uv run interventionfeatures discover discover.tasks=[ioi]

# Discover directions for RAVEL attributes
uv run interventionfeatures discover \
  discover.tasks=[ravel_country,ravel_continent,ravel_language]

# Find multiple directions per task
uv run interventionfeatures discover \
  discover.num_directions_per_task=3 \
  discover.orthogonality_weight=0.1

# Then evaluate discovered directions on benchmark
uv run interventionfeatures benchmark \
  benchmark.directions_path=task_directions.pkl
```

### Available Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| `ioi` | mib-bench/ioi | Indirect Object Identification |
| `arithmetic_add` | mib-bench/arithmetic_addition | Addition problems |
| `arithmetic_sub` | mib-bench/arithmetic_subtraction | Subtraction problems |
| `mcqa` | mib-bench/copycolors_mcqa | Multiple choice QA |
| `arc_easy` | mib-bench/arc_easy | ARC Easy questions |
| `arc_challenge` | mib-bench/arc_challenge | ARC Challenge questions |
| `ravel_country` | mib-bench/ravel | City → Country attribute |
| `ravel_language` | mib-bench/ravel | City → Language attribute |
| `ravel_continent` | mib-bench/ravel | City → Continent attribute |
| `ravel_timezone` | mib-bench/ravel | City → Timezone attribute |

### Configuration

```yaml
# conf/discover/default.yaml
tasks:
  - ioi
  - ravel_country

iterations: 512
batch_size: 32
learning_rate: 0.1
target_norm: 10.0
num_directions_per_task: 1
orthogonality_weight: 0.1
eval_freq: 50
```

### Comparison: Generic vs Task-Targeted CSS

| Aspect | Generic CSS | Task-Targeted CSS |
|--------|-------------|-------------------|
| Data | Random C4 samples | Task contrastive pairs |
| Objective | KL divergence (sensitivity) | IIA (task accuracy) |
| Output | Many generic features | Few task-specific features |
| Use case | Exploration | Benchmark submission |
| Efficiency | Find 10,000 features | Find 1-5 per task |

---

## RAVEL Benchmark

### Reference
- **Paper**: [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](https://arxiv.org/abs/2402.17700)
- **GitHub**: https://github.com/explanare/ravel
- **Dataset**: HuggingFace `hij/ravel`

### What it Measures

RAVEL evaluates how well a featurization method can **disentangle** entity-attribute representations. Given a feature that is supposed to encode a specific attribute (e.g., "country" for cities), RAVEL tests:

1. **Does intervening on the feature change the target attribute?** (CAUSE)
2. **Does intervening on the feature leave other attributes unchanged?** (Isolation)

### Metrics

#### CAUSE Score
Measures whether intervening on feature F changes the model's output for target attribute A.

**Formula:**
```
CAUSE(A, F, M, D) = E_D[II(M, F, x, x') = A_E']
```

- `II` = Interchange Intervention
- `x` = base input about entity E
- `x'` = source input about entity E'
- `A_E'` = attribute A's value for entity E'

**Interpretation**: When we swap the feature representation from entity E to E', does the model output E's attribute value instead of E's? Higher is better (1.0 = perfect).

**Example**:
- Base: "Paris is located in" → "France"
- Source: "Berlin is located in" → "Germany"
- After intervention (swap country feature): "Paris is located in" → "Germany"
- CAUSE = 1.0 if output is "Germany"

#### Isolation Score
Measures whether intervening on feature F does NOT change other attributes.

**Formula:**
```
ISO(A, F, M, D) = (1/|A\{A}|) Σ_{A* ∈ A\{A}} E_D[II(M, F, x*, x') = A*_E]
```

**Interpretation**: When we intervene on the feature for attribute A, do other attributes (A*) remain unchanged? Higher is better (1.0 = perfect isolation).

**Example**: If we intervene on the "country" feature for Paris→Berlin:
- "Paris speaks" should still output "French" (not "German")
- "Paris's timezone is" should remain unchanged

#### Disentangle Score
Combined metric:
```
Disentangle = (CAUSE + ISO) / 2
```

### Entity Types and Attributes

| Entity Type | # Entities | Attributes |
|-------------|------------|------------|
| Cities | 3,552 | country, language, latitude, longitude, timezone, continent |
| Nobel Laureates | 928 | field, year, country, gender, university |
| Verbs | 986 | tense, person, number, aspect |
| Physical Objects | 563 | color, size, material, shape |
| Occupations | 799 | sector, education, salary_range, work_environment |

### Usage

```bash
# Run RAVEL on all entity types
uv run interventionfeatures benchmark benchmark.enabled_benchmarks=[ravel]

# Run on specific entity types
uv run interventionfeatures benchmark \
  benchmark.enabled_benchmarks=[ravel] \
  benchmark.ravel.entity_types=[cities,nobel]

# Limit attributes per entity
uv run interventionfeatures benchmark \
  benchmark.enabled_benchmarks=[ravel] \
  benchmark.ravel.attributes_per_entity=2
```

---

## MIB Benchmark (Causal Variable Track)

### Reference
- **Paper**: [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151)
- **GitHub**: https://github.com/aaronmueller/mib
- **Datasets**: HuggingFace `mib-bench/*`
- **Leaderboard**: https://huggingface.co/spaces/mib-bench/leaderboard

### What it Measures

The MIB Causal Variable Track benchmarks **featurization methods**—methods that transform model activations into a space where it's easier to isolate a given causal variable. It tests whether features align with task-relevant causal variables using interchange interventions.

### Metric: Interchange Intervention Accuracy (IIA)

**Definition**: The proportion of examples where swapping feature values between base and source inputs causes the model output to match the source's expected output.

```
IIA = (# correct interventions) / (# total interventions)
```

**Interpretation**: Higher IIA means the feature better captures the causal variable. 1.0 = perfect alignment.

### Tasks

#### 1. IOI (Indirect Object Identification)

**What it tests**: Can the model identify the indirect object in sentences?

**Example**:
```
Base: "John gave Mary the book. Mary gave the book to ___"
Expected: "John"

Source: "Alice gave Bob the book. Bob gave the book to ___"
Expected: "Alice"
```

**Causal Variable**: The indirect object (IO) identity

**Intervention**: Swap the IO representation → model should output source's IO

**Dataset**: `mib-bench/ioi`

#### 2. Arithmetic (Addition/Subtraction)

**What it tests**: Can the model perform arithmetic and do features encode operands?

**Example (Addition)**:
```
Base: "23 + 45 = ___"  → Expected: "68"
Source: "23 + 12 = ___" → Expected: "35"
```

**Causal Variable**: The second operand value

**Intervention**: Swap operand representation → model should compute with source's operand

**Datasets**:
- `mib-bench/arithmetic_addition`
- `mib-bench/arithmetic_subtraction`

#### 3. MCQA (Multiple Choice Question Answering)

**What it tests**: Can the model answer questions about object properties?

**Example (CopyColors)**:
```
Base: "The apple is red. What color is the apple? A) red B) blue"
Expected: "A"

Source: "The apple is blue. What color is the apple? A) red B) blue"
Expected: "B"
```

**Causal Variable**: The object's color property

**Intervention**: Swap color representation → model should select source's answer

**Dataset**: `mib-bench/copycolors_mcqa`

#### 4. ARC (AI2 Reasoning Challenge)

**What it tests**: Science reasoning questions at different difficulty levels

**Example**:
```
Q: "Which property of air does a barometer measure?"
A) speed  B) pressure  C) humidity  D) temperature
Expected: "B"
```

**Difficulty Levels**:
- **Easy**: Grade-school level questions
- **Challenge**: More difficult questions requiring deeper reasoning

**Datasets**:
- `mib-bench/arc_easy`
- `mib-bench/arc_challenge`

#### 5. RAVEL (as MIB Task)

**What it tests**: City attribute disentanglement (same as standalone RAVEL, but using IIA metric)

**Example**:
```
Base: "Paris is located in ___" → Expected: "France"
Source: "Berlin is located in ___" → Expected: "Germany"
```

**Note**: This uses IIA for consistency with other MIB tasks, rather than RAVEL's native CAUSE/Isolation scores.

**Dataset**: `mib-bench/ravel`

### Usage

```bash
# Run all MIB tasks
uv run interventionfeatures benchmark benchmark.enabled_benchmarks=[mib]

# Run specific tasks
uv run interventionfeatures benchmark \
  benchmark.enabled_benchmarks=[mib] \
  benchmark.mib.tasks=[ioi,arithmetic_add]

# Limit samples per task (for faster testing)
uv run interventionfeatures benchmark \
  benchmark.enabled_benchmarks=[mib] \
  benchmark.mib.max_samples_per_task=100
```

---

## How CSS Directions are Evaluated

### The CSSFeaturizer

CSS directions are wrapped in a `CSSFeaturizer` class that provides:

1. **Forward transformation**: Project activations onto CSS direction space
   ```
   features = activations @ css_directions.T
   ```

2. **Inverse transformation**: Reconstruct activations from modified features
   ```
   new_activations = features @ css_directions + orthogonal_residual
   ```

3. **Interchange intervention**: Swap features between base and source
   ```python
   base_features = forward(base_activations)
   source_features = forward(source_activations)
   intervened_features = base_features.copy()
   intervened_features[dims_to_swap] = source_features[dims_to_swap]
   output = inverse(intervened_features, base_activations)
   ```

### Evaluation Pipeline

1. **Load CSS directions** from `directions.pkl`
2. **Create CSSFeaturizer** from directions
3. **For each benchmark example**:
   - Tokenize base and source inputs
   - Extract activations at intervention layer
   - Apply interchange intervention via featurizer
   - Run model with intervened activations
   - Check if output matches expected
4. **Aggregate scores** across examples

---

## Configuration Reference

### BenchmarkConfig

```yaml
benchmark:
  enabled_benchmarks: [ravel, mib]  # Which benchmarks to run
  directions_path: directions.pkl   # Path to CSS directions (relative to output_dir)
  direction_indices: null           # Which directions to use (null = all)
  orthonormalize_featurizer: true   # Orthonormalize CSS directions
  output_format: json               # Output format: json, pickle
```

### RAVELConfig

```yaml
benchmark:
  ravel:
    entity_types:                   # Entity types to evaluate
      - cities
      - nobel
      - verbs
      - objects
      - occupations
    attributes_per_entity: -1       # -1 = all attributes
    ravel_repo_path: null           # Optional path to cloned RAVEL repo
    use_cached_data: true
```

### MIBConfig

```yaml
benchmark:
  mib:
    tasks:                          # Tasks to evaluate
      - ioi
      - arithmetic_add
      - arithmetic_sub
      - mcqa
      - arc_easy
      - ravel
    max_samples_per_task: -1        # -1 = all samples
    hf_cache_dir: null              # HuggingFace cache directory
```

---

## Interpreting Results

### Output Format

Results are saved to `benchmark_results.json`:

```json
{
  "ravel": {
    "cities_country": {
      "task_name": "cities_country",
      "metrics": {
        "cause": 0.75,
        "isolation": 0.82,
        "disentangle": 0.785
      },
      "metadata": {
        "entity_type": "cities",
        "target_attribute": "country",
        "num_examples": 100
      }
    }
  },
  "mib": {
    "ioi": {
      "task_name": "ioi",
      "metrics": {
        "iia": 0.68
      },
      "metadata": {
        "num_samples": 500,
        "model": "pythia-70m-deduped",
        "layer": 2
      }
    }
  }
}
```

### What Good Scores Look Like

| Benchmark | Metric | Random Baseline | Good | Excellent |
|-----------|--------|-----------------|------|-----------|
| RAVEL | CAUSE | ~0.0 | >0.5 | >0.8 |
| RAVEL | Isolation | ~1.0 (trivial) | >0.7 | >0.9 |
| RAVEL | Disentangle | ~0.5 | >0.6 | >0.8 |
| MIB | IIA | Varies by task | >0.5 | >0.8 |

### Key Findings from Literature

From the MIB paper:
- **DAS (Distributed Alignment Search)** performs best on causal variable localization (supervised method)
- **SAE features are not consistently better than neurons** for these tasks
- Non-linear and low-dimensional projections can improve performance

From RAVEL:
- Most methods achieve higher Isolation than CAUSE
- Multi-task training improves disentanglement
- Smaller models tend to be more disentangled

---

## Troubleshooting

### Common Issues

**"directions.pkl not found"**
- Run `uv run interventionfeatures run` first to generate CSS directions

**"transformer_lens required"**
- Already included as a core dependency, but ensure you have: `uv sync`

**"pyvene required"**
- Install benchmark dependencies: `uv sync --extra benchmark`

**HuggingFace dataset loading fails**
- Check internet connection
- Set `HUGGINGFACE_API_KEY` in `.env` for private datasets
- Use `benchmark.mib.hf_cache_dir` to specify cache location

**Out of memory**
- Reduce `benchmark.mib.max_samples_per_task`
- Use fewer CSS directions: `benchmark.direction_indices=[0,1,2]`

---

## Experimental Results and Known Issues

### Results on Pythia-70m (December 2024)

Using 4 CSS directions found via generic CSS on C4 data:

#### MIB Results

| Task | IIA | Examples (after filter) | Notes |
|------|-----|------------------------|-------|
| **IOI** | 1.00 | 34 | Excellent - CSS captures IO identity |
| **MCQA** | 1.00 | 2 | **Problematic** - only 2 examples pass filter |
| **Arithmetic** | 0.00 | 0 | Model cannot do arithmetic at all |
| **Aggregate** | 0.67 | - | Misleading due to sample size issues |

#### RAVEL Results (Disentangle Scores)

| Entity Type | Best Attribute | Score | Worst Attribute | Score |
|-------------|---------------|-------|-----------------|-------|
| Cities | Timezone | 0.29 | Latitude | 0.04 |
| Nobel | Field | 0.25 | Award Year | 0.02 |
| Verbs | Pronunciation | 0.34 | - | - |
| Objects | Size | 0.49 | Category | 0.18 |
| Occupations | Industry | 0.06 | Duty | 0.00 |

**Average Disentangle: ~0.15-0.20**

### Known Issues

#### 1. Low Sample Count After Filtering (CRITICAL)

The MIB benchmark uses **canonical filtering**: only evaluate on examples where the model already performs correctly on both base and counterfactual inputs.

**Problem**: Pythia-70m (70M params) is too small for many tasks:

```
MCQA: 30 samples → 2 pass filter (93% filtered out)
Arithmetic: 30 samples → 0 pass filter (100% filtered out)
```

**Evidence**:
```python
# Model outputs for MCQA-style prompts:
"The color of the sky is A) red B) blue. Answer:" → " " (space)
"What is 2+2? A) 3 B) 4. Answer:" → "yes"
"Paris is in A) Germany B) France. Answer:" → "that"

# Model output for IOI (mib-bench/ioi format):
"Then, Henry and Phil had fun at the harbor. Henry gave a basket to"
→ " the harbor, and he" (expected: "Phil")
# Model continues text instead of predicting indirect object!
```

The model doesn't understand task formats and just continues text.

**Solution**:
- Use larger models (pythia-160m, pythia-410m) for meaningful benchmarking
- Use the MIB task modules (not raw mib-bench datasets) which use formats the model understands
- Focus on RAVEL which measures representation quality, not task performance

#### 2. CSS vs SAE Comparison

Published benchmarks using Llama2-7B (100x larger):

| Method | RAVEL Disentangle | Supervision |
|--------|-------------------|-------------|
| **MDAS** (SOTA) | 60-66% | Counterfactual |
| **DAS** | 56-57% | Counterfactual |
| **SAE** | 48.6% | None |
| **PCA** | 39.5% | None |
| **Our CSS (4 dirs)** | ~15-20% | None |

**Key differences**:
1. Model size: 7B vs 70M parameters
2. Feature count: SAEs have 16k-65k features, we have 4 directions
3. Layer: RAVEL peaks at layer ~15 in larger models

#### 3. Recommended Configuration

For honest benchmarking with Pythia-70m:

```bash
# Use IOI task (model can actually do it)
uv run interventionfeatures benchmark \
  benchmark=mib \
  benchmark.mib.tasks=[ioi] \
  benchmark.mib.max_samples_per_task=100

# Or use larger model
uv run interventionfeatures benchmark \
  model.model_name=EleutherAI/pythia-410m-deduped \
  model.layer_cutoff=8
```

### Why MIB Runner Got Better Results

The MIB benchmark runner uses the **MIB task module format** which loads data via:
```python
from tasks.IOI_task.ioi_task import get_counterfactual_datasets
```

This format is different from the raw `mib-bench/ioi` HuggingFace dataset. The task modules use prompts that the model can actually complete correctly, which is why the MIB runner showed 34 examples passing filter vs 1 example from the raw dataset.

### Comparison with SAE Baselines

EleutherAI provides pre-trained SAEs for Pythia models:
- [`EleutherAI/sae-pythia-70m-32k`](https://huggingface.co/EleutherAI/sae-pythia-70m-32k) - 32k latents per layer
- Uses [SAE-lens](https://github.com/jbloomAus/SAELens) or [EleutherAI/sae](https://github.com/EleutherAI/sae) library

**Quick comparison on IOI (mib-bench/ioi format, 50 samples)**:

| Method | IIA | Examples Evaluated | Notes |
|--------|-----|-------------------|-------|
| CSS (4 dirs) | 1.00 | 1 | Only 1/50 passed filter |
| SAE (32k latents, full swap) | 0.00 | 1 | SAE reconstruction lossy |
| Raw replacement | 1.00 | 1 | Baseline |

**Key insight**: With only 1 example passing filter, results are not statistically meaningful. The model is too small for this task format.

See `src/interventionfeatures/benchmarks/sae_runner.py` for SAE evaluation code.

---

## References

1. Huang et al. (2024). "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations." arXiv:2402.17700

2. Mueller et al. (2025). "MIB: A Mechanistic Interpretability Benchmark." ICML 2025. arXiv:2504.13151

3. Geiger et al. (2024). "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations." arXiv:2303.02536 (DAS method)

4. Wu et al. (2024). "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions." arXiv:2403.07809
