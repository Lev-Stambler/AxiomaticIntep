# Intervention Features

Mechanistic interpretability using CSS (Causal Scrubbing Search) and activation patching.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies and create venv
uv sync

# With optional features
uv sync --extra search     # Open-Puffer similarity search
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

### Open-Puffer Vector Database Setup

The pipeline uses [Open-Puffer](https://github.com/harishsg993010/open-puffer), a high-performance Rust-based vector database for similarity search.

**First-time setup: Build Open-Puffer (requires Rust)**

```bash
# Clone and build Open-Puffer
mkdir -p ~/.local/src
cd ~/.local/src
git clone https://github.com/harishsg993010/open-puffer.git
cd open-puffer

# Fix missing dependency (temporary upstream bug)
sed -i 's/parking_lot = { workspace = true }/parking_lot = { workspace = true }\nlibc = "0.2"/' crates/query/Cargo.toml

# Build in release mode
cargo build --release
```

The default config expects the binary at `~/.local/src/open-puffer/target/release/puffer-server`.

**Usage: Auto-start (Default)**

The server auto-starts when you run the pipeline:

```bash
uv run interventionfeatures run
```

**Usage: Manual Server Start**

Alternatively, start the server manually:

```bash
~/.local/src/open-puffer/target/release/puffer-server --bind-addr 0.0.0.0:8080 --data-dir ./data

# Then run with null binary path to connect to existing server
uv run interventionfeatures run database.openpuffer_binary_path=null
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

## Open-Puffer Similarity Search

The pipeline uses Open-Puffer to build an efficient similarity search index for finding high-activation examples.

### How It Works

**Index Building Process:**
1. During `run` command: extracts activations from ~10k samples (configurable)
2. Stores activation vectors in Open-Puffer with HNSW index
3. Saves token metadata for each activation
4. Persists index to `<output_dir>/puffer_data/`

**Search Process:**
- Two-stage search: fast approximate search → precise re-ranking
- Supports cosine similarity and dot product scoring
- Used for generating explanations and visualizations

**Performance:**
- Open-Puffer provides sub-2ms query latency (10x faster than alternatives)
- Index building: ~5-10 minutes for 10k samples (with batching optimization)
- Index reused across `explain` and `viz` commands

### Configuration

```yaml
# conf/database/default.yaml
index_size: 10000              # Number of samples to index
index_batch_size: 128          # Batch size (higher = faster)
scoring_type: "dot"            # "dot" or "cosine"
save_activations: true         # Cache activations to disk
use_MICS: true                 # Use MICS scoring

# Open-Puffer server settings (auto-start enabled by default)
openpuffer_binary_path: "${oc.env:HOME}/.local/src/open-puffer/target/release/puffer-server"
openpuffer_host: "localhost"
openpuffer_port: 8080
openpuffer_data_dir: null      # Defaults to output_dir/puffer_data
```

### Examples

```bash
# Build larger index
uv run python -m interventionfeatures.cli.main run database.index_size=50000

# Use cosine similarity
uv run python -m interventionfeatures.cli.main run database.scoring_type=cosine

# Use auto-start with binary path
uv run interventionfeatures run database.openpuffer_binary_path=/path/to/puffer-server

# Reuse existing index (specify output directory)
uv run python -m interventionfeatures.cli.main explain  # Uses index from Hydra's output dir
```

### Directory Structure

```
outputs/2025-12-16/14-30-45/
├── searcher_db/              # Token metadata
│   └── indexed_sample_tokens.pkl
├── puffer_data/              # Open-Puffer persistent storage
├── directions.pkl            # CSS directions
├── explanations.json         # LLM explanations
├── html/                     # Visualization data
│   └── *_visualization.json
└── config.yaml              # Run configuration
```

### Troubleshooting

**Error: "Open-Puffer server not available"**
- Ensure the Open-Puffer server is running on port 8080
- Or provide `database.openpuffer_binary_path` for auto-start

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
        search.py       # Similarity search (Open-Puffer)
        openpuffer_client.py  # Open-Puffer HTTP client
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

- `[search]` - requests library for Open-Puffer HTTP client
- `[explain]` - LangChain for AI-powered explanations
- `[benchmark]` - pyvene for RAVEL/MIB benchmarks
- `[dev]` - Testing and development tools

## License

MIT
