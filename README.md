# Intervention Features

Mechanistic interpretability using CSS (Causal Scrubbing Search) and activation patching.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies and create venv
uv sync

# With optional features
uv sync --extra search     # ChromaDB similarity search
uv sync --extra explain    # LangChain AI explanations
uv sync --extra benchmark  # RAVEL and MIB benchmarks
uv sync --extra dev        # Development tools
```

## Quick Start

### Environment Setup

Create a `.env` file with your API keys:

```bash
HUGGINGFACE_API_KEY=your_key
OPENAI_API_KEY=your_key  # for explanations
```

### CLI Usage

```bash
# Print default configuration
uv run interventionfeatures config

# Run CSS direction finding
uv run interventionfeatures run

# Override config values
uv run interventionfeatures run model.layer_cutoff=3 training.dict_size=5

# Generate explanations for found directions
uv run interventionfeatures explain results.pkl db_path/

# Run benchmark evaluations (RAVEL and MIB)
uv run interventionfeatures benchmark
```

### Configuration

Configuration uses Hydra with YAML files in `conf/`:

```
conf/
├── config.yaml       # Main config with defaults
├── model/            # Model presets
├── training/         # Training presets
├── database/         # Database presets
├── llm/              # LLM presets
├── dataset/          # Dataset presets
└── benchmark/        # Benchmark presets (RAVEL, MIB)
```

Override any value via CLI:

```bash
uv run interventionfeatures run model.model_name="EleutherAI/pythia-160m" seed=42
```

## Benchmarks

The package includes support for standard mechanistic interpretability benchmarks:

### RAVEL

[RAVEL](https://arxiv.org/abs/2402.17700) evaluates feature disentanglement using interchange interventions.

**Metrics:**
- **CAUSE Score**: Measures if intervening on feature F changes target attribute A
- **Isolation Score**: Measures if interventions on F don't affect other attributes
- **Disentangle Score**: (CAUSE + Isolation) / 2

**Entity Types:** Cities, Nobel Laureates, Verbs, Physical Objects, Occupations

### MIB Causal Variable Track

[MIB](https://arxiv.org/abs/2504.13151) benchmarks featurization methods on causal variable localization.

**Metric:** Interchange Intervention Accuracy (IIA)

**Tasks:**
- IOI (Indirect Object Identification)
- Arithmetic (Addition/Subtraction)
- MCQA (Multiple Choice QA)
- ARC (AI2 Reasoning Challenge)
- RAVEL

### Benchmark Usage

```bash
# Run all benchmarks
uv run interventionfeatures benchmark

# RAVEL only on cities
uv run interventionfeatures benchmark \
  benchmark.enabled_benchmarks=[ravel] \
  benchmark.ravel.entity_types=[cities]

# MIB IOI task only
uv run interventionfeatures benchmark \
  benchmark.enabled_benchmarks=[mib] \
  benchmark.mib.tasks=[ioi]

# Use specific CSS directions
uv run interventionfeatures benchmark benchmark.direction_indices=[0,1,2]
```

## Development

```bash
uv run pytest              # Run tests
uv run ruff check src/     # Lint
uv run ruff format src/    # Format
```

## Package Structure

```
src/interventionfeatures/
    config.py           # Hydra configuration
    core/
        css.py          # CSS direction finding
        data_handler.py # Data handling
        model.py        # Model interventions
        search.py       # Similarity search
    analysis/
        explainer.py    # Feature explanation
        faithfulness.py # Faithfulness testing
    benchmarks/
        featurizer.py   # CSS direction wrapper for benchmarks
        base.py         # Base benchmark runner
        ravel/          # RAVEL benchmark
        mib/            # MIB benchmark (5 tasks)
    cli/
        main.py         # CLI entry point
```

## Optional Dependencies

- `[search]` - ChromaDB for similarity search
- `[explain]` - LangChain for AI-powered explanations
- `[benchmark]` - pyvene for RAVEL/MIB benchmarks
- `[dev]` - Testing and development tools

## License

MIT
