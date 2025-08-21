# Makefile for interventionfeatures project

.PHONY: help install install-dev test test-fast lint format type-check clean build docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-fast    - Run fast tests only (exclude slow tests)"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run type checking with mypy"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Generate documentation"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing targets
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-coverage:
	pytest tests/ --cov=src/interventionfeatures --cov-report=html --cov-report=term

# Code quality targets
lint:
	flake8 src/ tests/
	pylint src/interventionfeatures/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:
	mypy src/interventionfeatures/

# Development targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

docs:
	sphinx-build -b html docs/ docs/_build/html

# Run all quality checks
check-all: format-check lint type-check test

# Development workflow
dev-setup: install-dev
	pre-commit install

# Quick development test
dev-test: format lint test-fast
