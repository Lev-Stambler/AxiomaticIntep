# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Installation
```bash
# Basic installation
pip install -e .

# Development installation with all dependencies
pip install -e ".[dev]"

# Install with specific features
pip install -e ".[all]"          # All optional features
pip install -e ".[search]"       # ChromaDB similarity search
pip install -e ".[explain]"      # AI explanations with LangChain
pip install -e ".[optimization]" # Hyperparameter tuning with Ax
pip install -e ".[gui]"          # GUI labeling interface
```

### Professional CLI Usage
```bash
# Generate configuration file
interventionfeatures config --output my_config.toml

# Run the full pipeline
interventionfeatures run --model-name "EleutherAI/pythia-70m-deduped" --layer-cutoff 2

# Run with config file
interventionfeatures run --config-file my_config.toml

# Generate explanations for existing results
interventionfeatures explain css_results.pkl searcher_db --gui

# Validate explanations
interventionfeatures validate css_results.pkl explanations.json
```

### Examples and Testing
```bash
# Jupyter notebook examples
jupyter notebook examples/test-pipeline.ipynb      # Main testing/example notebook
jupyter notebook examples/basic-usage.ipynb        # Basic usage examples
jupyter notebook examples/advanced-usage.ipynb     # Advanced CLI usage examples

# Test pattern references
python examples/test-patterns.py                   # Testing patterns and mock data
python examples/test_activation_patcher.py         # Activation patching tests
python examples/test_early_stopping.py             # Early stopping tests
python examples/test_human_labeling.py             # Human labeling workflow tests
```

### Testing
```bash
# Run all tests
make test
pytest tests/ -v

# Run fast tests only (exclude slow tests)
make test-fast
pytest tests/ -v -m "not slow"

# Run with coverage
make test-coverage
pytest tests/ --cov=src/interventionfeatures --cov-report=html --cov-report=term

# Additional test examples (see examples/ directory)
python examples/test_activation_patcher.py
python examples/test_early_stopping.py
```

### Code Quality
```bash
# Format code
make format
black src/ tests/
isort src/ tests/

# Lint code
make lint
flake8 src/ tests/
pylint src/interventionfeatures/

# Type check
make type-check
mypy src/interventionfeatures/

# Run all quality checks
make check-all
```

### Development Setup
```bash
# Set up development environment
make dev-setup
pip install -e ".[dev]"
pre-commit install

# Quick development test
make dev-test
```

### Documentation
```bash
# View detailed guides
docs/early-stopping.md     # Early stopping implementation guide
docs/human-labeling.md     # Human labeling workflow guide
docs/project-notes.md      # Project insights and TODOs
```

## Code Architecture

This repository contains a professional Python package (`src/interventionfeatures/`) with examples (`examples/`) and documentation (`docs/`).

### Package Structure (`src/interventionfeatures/`)

#### Core Modules (`core/`)
- **`css.py`** - `CSSDirectionFinder` class implementing Causal Scrubbing Search with PGA optimization
- **`data_handler.py`** - `TransformerDataHandler` for memory-efficient activation processing
- **`model.py`** - `IntervenableTransformerSegment` for targeted model interventions
- **`sae.py`** - Sparse Autoencoder utilities and training
- **`search.py`** - `ActivationSimSearcher` with ChromaDB vector similarity search
- **`scheduler.py`** - Training schedulers and optimization utilities

#### Analysis Modules (`analysis/`)
- **`explainer.py`** - AI-powered explanation generation and human labeling interface
- **`faithfulness.py`** - Faithfulness evaluation for interventions and explanations
- **`gui/`** - GUI components for manual labeling and visualization

#### Utilities (`utils/`)
- **`config.py`** - `MainConfig` class with type-safe configuration management
- **`math_utils.py`** - Mathematical utilities for tensor operations
- **`visualization.py`** - Plotting and HTML visualization generation

#### CLI (`cli/`)
- **`main.py`** - Professional CLI interface using Typer with rich formatting

### Examples and Documentation Structure
- **`examples/`** - Jupyter notebooks and test patterns demonstrating usage
  - `test-pipeline.ipynb` - Main testing/example notebook
  - `basic-usage.ipynb` - Basic usage examples
  - `advanced-usage.ipynb` - Advanced CLI usage examples
  - `test_*.py` - Test patterns for various components
- **`docs/`** - Detailed documentation and guides
  - `early-stopping.md` - Early stopping implementation guide
  - `human-labeling.md` - Human labeling workflow guide
  - `project-notes.md` - Project insights and development notes

### Key Architectural Patterns

#### Configuration Management
- All parameters managed through `MainConfig` class with Pydantic validation
- Support for JSON/TOML configuration files
- Environment-specific overrides available

#### Optional Dependencies
The package uses optional dependencies to keep core lightweight:
- ChromaDB for similarity search (`[search]`)
- LangChain + Anthropic for AI explanations (`[explain]`)
- Ax Platform for hyperparameter optimization (`[optimization]`)
- tkinter for GUI features (`[gui]`)

#### Data Flow Architecture
1. **Configuration** → `MainConfig` loads parameters
2. **Data Loading** → `TransformerDataHandler` processes datasets
3. **Model Setup** → `IntervenableTransformerSegment` wraps transformers
4. **Direction Finding** → `CSSDirectionFinder` uses PGA optimization
5. **Similarity Search** → `ActivationSimSearcher` queries ChromaDB
6. **Analysis** → Explanation generation and faithfulness testing
7. **Visualization** → HTML output with interactive features

### Research Domain: Mechanistic Interpretability

This codebase implements techniques for understanding neural networks mechanistically:

- **Causal Scrubbing Search (CSS)** - Method for finding causally relevant directions
- **Activation Patching** - Targeted interventions to understand model behavior
- **Sparse Autoencoders (SAEs)** - Learning interpretable feature representations
- **Fourier Features** - Using frequency domain analysis for interpretability
- **Faithfulness Testing** - Validating explanations through automated testing

### Testing Strategy
- **Unit tests** - Individual component testing
- **Integration tests** - End-to-end pipeline testing
- **Slow tests** - Marked separately for CI optimization
- **Coverage reporting** - Comprehensive test coverage tracking

### Data Directories
- `data-*/` - Generated experiment results with parameter hash identifiers
- Multiple output formats: HTML visualizations, JSON results, pickle files
- ChromaDB databases stored in `db-*/` directories

## Conda and Development Notes

- We are using conda and $CONDA_ENV should be used for pip
