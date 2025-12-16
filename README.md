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
OPENROUTER_API_KEY=your_key  # for explanations (recommended)
# or
OPENAI_API_KEY=your_key  # alternative to OpenRouter
```

### CLI Usage

```bash
# Print default configuration
uv run interventionfeatures config

# Run CSS direction finding (includes visualization generation)
uv run python -m interventionfeatures.cli.main run dataset.dataset_config=en

# Override config values
uv run python -m interventionfeatures.cli.main run model.layer_cutoff=3 training.dict_size=5

# Task-targeted CSS discovery (optimizes directly for benchmark IIA)
uv run interventionfeatures discover discover.tasks=[ioi,ravel_country]

# Generate LLM explanations (requires OPENROUTER_API_KEY or OPENAI_API_KEY)
uv run python -m interventionfeatures.cli.main explain llm=openrouter

# View visualizations in Streamlit
uv run python -m interventionfeatures.cli.main viz outputs/2025-12-16/14-30-45/

# Run benchmark evaluations (RAVEL and MIB)
uv run python -m interventionfeatures.cli.main benchmark
```

**Note:** Outputs are automatically saved to timestamped directories: `outputs/YYYY-MM-DD/HH-MM-SS/`

## ChromaDB Similarity Search

The pipeline uses ChromaDB to build an efficient similarity search index for finding high-activation examples.

### How It Works

**Index Building Process:**
1. During `run` command: extracts activations from ~10k samples (configurable)
2. Stores activation vectors in ChromaDB with HNSW index
3. Saves token metadata for each activation
4. Persists index to `<output_dir>/searcher_db/`

**Search Process:**
- Two-stage search: fast approximate search → precise re-ranking
- Supports cosine similarity and dot product scoring
- Used for generating explanations and visualizations

**Performance:**
- Index building: ~5-10 minutes for 10k samples (with batching optimization)
- Search queries: <1 second
- Index reused across `explain` and `viz` commands

### Configuration

```yaml
# conf/database/default.yaml
index_size: 10000          # Number of samples to index
index_batch_size: 128      # Batch size (higher = faster)
scoring_type: "dot"        # "dot" or "cosine"
save_activations: true     # Cache activations to disk
```

### Examples

```bash
# Build larger index
uv run python -m interventionfeatures.cli.main run database.index_size=50000

# Use cosine similarity
uv run python -m interventionfeatures.cli.main run database.scoring_type=cosine

# Reuse existing index (specify output directory)
uv run python -m interventionfeatures.cli.main explain  # Uses index from Hydra's output dir
```

### Directory Structure

```
outputs/2025-12-16/14-30-45/
├── searcher_db/              # ChromaDB persistent storage
│   ├── chroma.sqlite3        # Metadata database
│   └── indexed_sample_tokens.pkl
├── directions.pkl            # CSS directions
├── explanations.json         # LLM explanations
├── html/                     # Visualization data
│   └── *_visualization.json
└── config.yaml              # Run configuration
```

### Troubleshooting

**Error: "ChromaDB not indexed"**
- Run `uv run interventionfeatures run` first to build the index

**Slow index building**
- Increase `database.index_batch_size` to 256 or 512
- Reduce `database.index_size` for faster iteration

**Out of memory during indexing**
- Decrease `database.index_batch_size`
- Set `database.save_activations=false` to reduce memory usage

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
