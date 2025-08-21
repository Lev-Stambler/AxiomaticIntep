#!/usr/bin/env python3
"""
TOML to Results Directory Mapping Generator

This script scans TOML configuration files and matches them to their corresponding
result directories based on parameter hashes. It generates a comprehensive JSON
mapping that shows which TOML files correspond to which output directories.

Usage:
    python scripts/generate_toml_mapping.py --output toml_to_results_mapping.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add the src directory to Python path to import the project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import only the config module to avoid dependency issues
try:
    from interventionfeatures.utils.config import MainConfig, generate_param_hash
except ImportError:
    # Fallback: import directly from the file
    import importlib.util
    config_path = Path(__file__).parent.parent / "src" / "interventionfeatures" / "utils" / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    MainConfig = config_module.MainConfig
    generate_param_hash = config_module.generate_param_hash


def find_toml_files(base_dir: Path) -> List[Path]:
    """Find all TOML files in the configs directory."""
    toml_files = []
    configs_dir = base_dir / "configs"
    
    if not configs_dir.exists():
        print(f"Warning: configs directory not found at {configs_dir}")
        return toml_files
    
    for toml_file in configs_dir.rglob("*.toml"):
        # Skip any files that might be temporary or backup files
        if not toml_file.name.startswith('.') and not toml_file.name.endswith('~'):
            toml_files.append(toml_file)
    
    return sorted(toml_files)


def load_toml_config(toml_path: Path) -> Optional[MainConfig]:
    """Load a TOML configuration file, return None if failed."""
    try:
        return MainConfig.from_toml_file(toml_path)
    except Exception as e:
        print(f"Warning: Failed to load {toml_path}: {e}")
        return None


def get_expected_hashes(config: MainConfig) -> Dict[str, str]:
    """Get the expected directory hashes for a configuration."""
    hashes = {}
    
    # Hash for directions/results directories (based on data params)
    data_params = config.get_data_params_dict()
    data_hash = generate_param_hash(data_params)
    hashes['data'] = data_hash
    
    # Hash for search directories (based on db params)  
    db_params = config.get_db_params_dict()
    db_hash = generate_param_hash(db_params)
    hashes['search'] = db_hash
    
    return hashes


def scan_output_directories(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan the outputs directory and collect information about existing directories."""
    outputs_dir = base_dir / "outputs"
    directories = {}
    
    if not outputs_dir.exists():
        print(f"Warning: outputs directory not found at {outputs_dir}")
        return directories
    
    for dir_path in outputs_dir.iterdir():
        if not dir_path.is_dir():
            continue
            
        dir_name = dir_path.name
        
        # Extract directory type and hash
        if '_' not in dir_name:
            continue
            
        parts = dir_name.split('_', 1)
        if len(parts) != 2:
            continue
            
        dir_type, hash_part = parts
        
        # Collect metadata about the directory (store relative path)
        relative_path = dir_path.relative_to(base_dir)
        dir_info = {
            'path': str(relative_path),
            'type': dir_type,
            'hash': hash_part,
            'exists': True,
            'files': {},
            'metadata': None,
            'timestamp': None,
            'completeness_score': 0
        }
        
        # Check for key files and metadata
        key_files = {
            'directions.pkl': 'directions_file',
            'explanations.json': 'explanations_file', 
            'metadata.json': 'metadata_file',
            'index.html': 'index_file',
            'validation_results.json': 'validation_file'
        }
        
        for filename, file_key in key_files.items():
            file_path = dir_path / filename
            dir_info['files'][file_key] = file_path.exists()
            if file_path.exists():
                dir_info['completeness_score'] += 1
        
        # Check for HTML visualizations directory
        html_dir = dir_path / "html"
        if html_dir.exists() and html_dir.is_dir():
            html_files = list(html_dir.glob("*.html"))
            dir_info['files']['html_visualizations'] = len(html_files)
            dir_info['completeness_score'] += min(len(html_files), 5)  # Cap at 5 points
        else:
            dir_info['files']['html_visualizations'] = 0
        
        # Load metadata if available
        metadata_file = dir_path / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    dir_info['metadata'] = metadata
                    if 'timestamp' in metadata:
                        dir_info['timestamp'] = metadata['timestamp']
            except Exception as e:
                print(f"Warning: Failed to load metadata from {metadata_file}: {e}")
        
        # For search directories, check for ChromaDB files
        if dir_type == 'search':
            chroma_db = dir_path / "chroma.sqlite3"
            if chroma_db.exists():
                dir_info['completeness_score'] += 2
                dir_info['files']['chroma_db'] = True
            else:
                dir_info['files']['chroma_db'] = False
        
        directories[dir_name] = dir_info
    
    return directories


def match_tomls_to_directories(
    toml_files: List[Path], 
    directories: Dict[str, Dict[str, Any]],
    base_dir: Path
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Match TOML files to their corresponding directories."""
    mappings = {}
    orphaned_directories = []
    matched_hashes = set()
    
    for toml_path in toml_files:
        # Get relative path from base directory
        try:
            relative_path = toml_path.relative_to(base_dir)
        except ValueError:
            relative_path = toml_path
        
        toml_key = str(relative_path)
        
        # Load the configuration
        config = load_toml_config(toml_path)
        if config is None:
            mappings[toml_key] = {
                'status': 'error',
                'error': 'Failed to load TOML file',
                'directories': {},
                'hashes': {}
            }
            continue
        
        # Get expected hashes
        expected_hashes = get_expected_hashes(config)
        
        # Find matching directories
        matching_dirs = {}
        status_parts = []
        
        # Look for directions and results directories (same hash)
        data_hash = expected_hashes['data']
        directions_key = f"directions_{data_hash}"
        results_key = f"results_{data_hash}"
        
        if directions_key in directories:
            matching_dirs['directions'] = directories[directions_key]['path']
            matched_hashes.add(data_hash)
            status_parts.append('directions')
        
        if results_key in directories:
            matching_dirs['results'] = directories[results_key]['path']
            matched_hashes.add(data_hash)
            status_parts.append('results')
        
        # Look for search directory
        search_hash = expected_hashes['search']
        search_key = f"search_{search_hash}"
        
        if search_key in directories:
            matching_dirs['search'] = directories[search_key]['path']
            matched_hashes.add(search_hash)
            status_parts.append('search')
        
        # Determine status and get latest timestamp
        if not matching_dirs:
            status = 'no_results'
            last_updated = None
        else:
            status = 'partial' if len(status_parts) < 3 else 'complete'
            if 'results' in matching_dirs and 'search' in matching_dirs:
                status = 'complete'
            elif 'results' in matching_dirs or ('directions' in matching_dirs and 'search' in matching_dirs):
                status = 'partial'
            else:
                status = 'minimal'
        
        # Get the most recent timestamp from matching directories
        timestamps = []
        for dir_type in ['directions', 'results', 'search']:
            if dir_type in matching_dirs:
                dir_key = f"{dir_type}_{expected_hashes['data'] if dir_type != 'search' else expected_hashes['search']}"
                if dir_key in directories and directories[dir_key]['timestamp']:
                    timestamps.append(directories[dir_key]['timestamp'])
        
        last_updated = max(timestamps) if timestamps else None
        
        # Extract key configuration parameters for easy reference
        config_summary = {
            'model_name': config.model_name,
            'layer_cutoff': config.layer_cutoff,
            'target_layers': config.target_layers,
            'target_token_offset': config.target_token_offset,
            'dict_size': config.dict_size
        }
        
        mappings[toml_key] = {
            'config_summary': config_summary,
            'hashes': expected_hashes,
            'directories': matching_dirs,
            'status': status,
            'last_updated': last_updated,
            'completeness_details': status_parts
        }
    
    # Find orphaned directories (those not matched to any TOML)
    for dir_name, dir_info in directories.items():
        if dir_info['hash'] not in matched_hashes:
            orphaned_directories.append(dir_name)
    
    return mappings, orphaned_directories


def calculate_summary_stats(
    mappings: Dict[str, Dict[str, Any]], 
    directories: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate summary statistics for the mapping."""
    stats = {
        'total_tomls': len(mappings),
        'total_directories': len(directories),
        'tomls_with_results': 0,
        'tomls_complete': 0,
        'tomls_partial': 0,
        'tomls_no_results': 0,
        'directories_by_type': defaultdict(int),
        'orphaned_count': 0
    }
    
    for toml_data in mappings.values():
        status = toml_data['status']
        if status == 'complete':
            stats['tomls_complete'] += 1
            stats['tomls_with_results'] += 1
        elif status in ['partial', 'minimal']:
            stats['tomls_partial'] += 1
            stats['tomls_with_results'] += 1
        elif status == 'no_results':
            stats['tomls_no_results'] += 1
    
    for dir_info in directories.values():
        stats['directories_by_type'][dir_info['type']] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate mapping from TOML configuration files to result directories"
    )
    parser.add_argument(
        "--output", "-o",
        default="toml_to_results_mapping.json",
        help="Output JSON file path (default: toml_to_results_mapping.json)"
    )
    parser.add_argument(
        "--base-dir", "-b",
        type=Path,
        default=Path.cwd(),
        help="Base directory of the project (default: current directory)"
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print the JSON output"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    base_dir = args.base_dir.resolve()
    
    if args.verbose:
        print(f"Scanning project directory: {base_dir}")
    
    # Find all TOML files
    toml_files = find_toml_files(base_dir)
    if args.verbose:
        print(f"Found {len(toml_files)} TOML files:")
        for toml_file in toml_files:
            print(f"  - {toml_file.relative_to(base_dir)}")
    
    # Scan output directories
    directories = scan_output_directories(base_dir)
    if args.verbose:
        print(f"Found {len(directories)} output directories:")
        for dir_name in sorted(directories.keys()):
            dir_info = directories[dir_name]
            print(f"  - {dir_name} (completeness: {dir_info['completeness_score']})")
    
    # Match TOMLs to directories
    mappings, orphaned_directories = match_tomls_to_directories(toml_files, directories, base_dir)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_stats(mappings, directories)
    
    # Create the final mapping structure
    result = {
        'toml_mappings': mappings,
        'orphaned_directories': sorted(orphaned_directories),
        'summary_statistics': summary_stats,
        'generated_at': datetime.now().isoformat(),
        'base_directory': str(base_dir)
    }
    
    # Write the output file
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    
    indent = 2 if args.pretty else None
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=indent, default=str)
    
    print(f"Generated mapping file: {output_path}")
    print(f"Summary:")
    print(f"  - Total TOML files: {summary_stats['total_tomls']}")
    print(f"  - TOMLs with results: {summary_stats['tomls_with_results']}")
    print(f"  - Complete: {summary_stats['tomls_complete']}")
    print(f"  - Partial: {summary_stats['tomls_partial']}")
    print(f"  - No results: {summary_stats['tomls_no_results']}")
    print(f"  - Orphaned directories: {len(orphaned_directories)}")
    print(f"  - Total directories: {summary_stats['total_directories']}")
    
    if orphaned_directories and args.verbose:
        print(f"Orphaned directories:")
        for orphan in sorted(orphaned_directories):
            print(f"  - {orphan}")


if __name__ == "__main__":
    main()
