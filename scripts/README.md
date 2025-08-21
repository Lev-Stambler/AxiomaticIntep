# Scripts Directory

This directory contains utility scripts for the FourierFeaturesIsMechInterp project.

## generate_toml_mapping.py

A Python script that creates a comprehensive mapping between TOML configuration files and their corresponding result directories.

### Purpose

The project generates result directories with hash-based names (e.g., `results_1479ac0bb7b3`, `directions_d404985edc9d`) based on configuration parameters. This script helps you:

- Find which TOML configuration corresponds to which result directories
- Identify orphaned directories (results without matching TOMLs)
- Get an overview of experiment completeness and status
- Track the most recent runs for each configuration

### Usage

```bash
# Generate mapping with default settings
python scripts/generate_toml_mapping.py

# Generate with custom output file and pretty formatting
python scripts/generate_toml_mapping.py --output my_mapping.json --pretty

# Run with verbose output to see detailed progress
python scripts/generate_toml_mapping.py --verbose

# Specify a different base directory
python scripts/generate_toml_mapping.py --base-dir /path/to/project
```

### Output Format

The script generates a JSON file with the following structure (note that all directory paths are relative to the project root):

```json
{
  "toml_mappings": {
    "configs/pythia-70m-0-offset-output/layer_1.toml": {
      "config_summary": {
        "model_name": "EleutherAI/pythia-70m-deduped",
        "layer_cutoff": 1,
        "target_token_offset": 0,
        "dict_size": 10
      },
      "hashes": {
        "data": "d404985edc9d",
        "search": "23ade17a0e36"
      },
      "directories": {
        "directions": "outputs/directions_d404985edc9d",
        "results": "outputs/results_d404985edc9d"
      },
      "status": "partial",
      "last_updated": "2025-07-31T23:03:27.471401",
      "completeness_details": ["directions", "results"]
    }
  },
  "orphaned_directories": ["results_a38075e6b854", ...],
  "summary_statistics": {
    "total_tomls": 14,
    "tomls_with_results": 7,
    "tomls_complete": 2,
    "tomls_partial": 5,
    "tomls_no_results": 7,
    "orphaned_count": 52
  }
}
```

### Status Meanings

- **complete**: Has results and search directories
- **partial**: Has directions and/or results but missing search directory
- **minimal**: Has only directions directory
- **no_results**: No matching result directories found

### Directory Types

- **directions_***: Raw direction finding results from CSS algorithm
- **results_***: Processed results with explanations and visualizations  
- **search_***: ChromaDB similarity search indexes

The script automatically matches TOML configurations to these directories based on parameter hashes computed from the configuration content.