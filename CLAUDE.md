# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Development Commands

```bash
# Install dependencies
uv sync

# Install with optional features
uv sync --extra dev      # Development tools
uv sync --extra search   # Open-Puffer HTTP client (requests)
uv sync --extra explain  # LangChain
```

## Open-Puffer Setup

The vector database uses Open-Puffer (Rust-based). Build it once:

```bash
mkdir -p ~/.local/src && cd ~/.local/src
git clone https://github.com/harishsg993010/open-puffer.git && cd open-puffer
sed -i 's/parking_lot = { workspace = true }/parking_lot = { workspace = true }\nlibc = "0.2"/' crates/query/Cargo.toml
cargo build --release
```

The server auto-starts when running the pipeline.

## CLI Usage

```bash
# Print default configuration
uv run interventionfeatures config

# Run the CSS direction finding pipeline
uv run interventionfeatures run

# Override config values
uv run interventionfeatures run model.layer_cutoff=3 training.dict_size=5

# Generate explanations
uv run interventionfeatures explain results.pkl db_path/
```

## Testing

```bash
uv run pytest tests/         # Run tests
uv run ruff check src/       # Lint
uv run ruff format src/      # Format
```

## Code Architecture

### Package Structure (`src/interventionfeatures/`)

```
src/interventionfeatures/
    __init__.py        # Package exports
    config.py          # Hydra dataclass configuration

    core/
        css.py         # CSSDirectionFinder - main algorithm
        data_handler.py # TransformerDataHandler
        model.py       # IntervenableTransformerSegment
        search.py      # ActivationSimSearcher (Open-Puffer)
        openpuffer_client.py  # Open-Puffer HTTP client
        scheduler.py   # Learning rate schedulers

    analysis/
        explainer.py      # FeatureExplainer (LangChain)
        faithfulness.py   # Faithfulness testing

    utils/
        math_utils.py     # Mathematical utilities
        visualization.py  # HTML visualization

    cli/
        main.py           # Hydra-based CLI
```

### Configuration

Configuration uses Hydra with YAML files in `conf/`:

```
conf/
    config.yaml       # Main config with defaults
    model/
        pythia-70m.yaml
    training/
        default.yaml
    database/
        default.yaml
    llm/
        openai.yaml
    dataset/
        c4.yaml
```

Override any config value via CLI:

```bash
uv run interventionfeatures run model.layer_cutoff=5 training.learning_rate=0.1
```

### Optional Dependencies

- `[search]` - requests for Open-Puffer HTTP client
- `[explain]` - LangChain for AI explanations
- `[dev]` - Testing and linting tools

## Extra for Claude
Don't run any tests locally, ask for the runpod instance of ssh instance
