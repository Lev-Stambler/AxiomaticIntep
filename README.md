# Intervention Features

Mechanistic interpretability using CSS (Causal Scrubbing Search) and activation patching.

## Installation

Requires Python 3.10+

```bash
# Using uv (recommended)
uv pip install -e .

# With dev dependencies
uv pip install -e ".[dev]"

# With all optional features
uv pip install -e ".[all]"
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
# Run CSS direction finding
interventionfeatures run

# Override config values
interventionfeatures run model.layer_cutoff=3 training.dict_size=5

# Generate explanations for found directions
interventionfeatures explain

# Print default configuration
interventionfeatures config
```

### Configuration

Configuration uses Hydra with YAML files in `conf/`:

```
conf/
    config.yaml       # Main config
    model/            # Model configs
    training/         # Training configs
    database/         # Database configs
    llm/              # LLM configs
    dataset/          # Dataset configs
```

Override any value via CLI:

```bash
interventionfeatures run model.model_name="EleutherAI/pythia-160m"
```

## Development

```bash
make test     # Run tests
make lint     # Run linting
make format   # Format code
make clean    # Clean build artifacts
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
