# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Development Commands

```bash
# Install with uv
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install with all optional features
uv pip install -e ".[all]"
```

## CLI Usage

```bash
# Run the CSS direction finding pipeline
interventionfeatures run

# Override config values
interventionfeatures run model.layer_cutoff=3 training.dict_size=5

# Generate explanations
interventionfeatures explain

# Print default configuration
interventionfeatures config
```

## Testing

```bash
make test        # Run all tests
make lint        # Run linting
make format      # Format code
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
        search.py      # ActivationSimSearcher (ChromaDB)
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
interventionfeatures run model.layer_cutoff=5 training.learning_rate=0.1
```

### Optional Dependencies

- `[search]` - ChromaDB for similarity search
- `[explain]` - LangChain for AI explanations
- `[dev]` - Testing and linting tools
