# Intervention Features

Mechanistic interpretability using CSS (Causal Scrubbing Search) and activation patching.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies and create venv
uv sync

# With optional features
uv sync --extra search   # ChromaDB similarity search
uv sync --extra explain  # LangChain AI explanations
uv sync --extra dev      # Development tools
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
└── dataset/          # Dataset presets
```

Override any value via CLI:

```bash
uv run interventionfeatures run model.model_name="EleutherAI/pythia-160m" seed=42
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
    cli/
        main.py         # CLI entry point
```

## Optional Dependencies

- `[search]` - ChromaDB for similarity search
- `[explain]` - LangChain for AI-powered explanations
- `[dev]` - Testing and development tools

## License

MIT
