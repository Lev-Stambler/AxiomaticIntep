#!/usr/bin/env python3
"""
Generate a master index HTML page for all experiment results.

This script reads the TOML to results mapping JSON and creates a comprehensive
index page with cards for each experiment, allowing easy navigation to individual
result visualizations.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_mapping_data(mapping_path: Path) -> Dict[str, Any]:
    """Load the TOML to results mapping JSON."""
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in mapping file: {e}")
        raise


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable string."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return timestamp_str


def get_status_badge_class(status: str) -> str:
    """Get CSS class for status badge."""
    if status == "complete":
        return "status-complete"
    elif status == "partial":
        return "status-partial"
    else:
        return "status-unknown"


def check_file_exists(base_path: Path, relative_path: str) -> bool:
    """Check if a file exists relative to base path."""
    return (base_path / relative_path).exists()


def generate_experiment_cards(mapping_data: Dict[str, Any], base_path: Path) -> List[str]:
    """Generate HTML cards for each experiment."""
    cards = []
    
    for toml_path, experiment in mapping_data.get("toml_mappings", {}).items():
        config = experiment.get("config_summary", {})
        directories = experiment.get("directories", {})
        status = experiment.get("status", "unknown")
        last_updated = experiment.get("last_updated", "")
        
        # Check if index.html exists for this experiment
        results_dir = directories.get("results", "")
        index_path = f"{results_dir.replace('outputs/', '')}/index.html" if results_dir else ""
        has_index = check_file_exists(base_path, results_dir + '/index.html') if index_path else False
        
        # Extract key configuration details
        model_name = config.get("model_name", "Unknown")
        layer_cutoff = config.get("layer_cutoff", "?")
        target_layer = config.get("target_layers")[0]
        token_offset = config.get("target_token_offset", "?")
        dict_size = config.get("dict_size", "?")
        
        # Create readable experiment name
        experiment_name = f"$a = {layer_cutoff}, b = {target_layer}, t = {token_offset}$"
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Status badge
        status_class = get_status_badge_class(status)
        status_text = status.title()
        
        # Format timestamp
        formatted_time = format_timestamp(last_updated) if last_updated else "Unknown"
        
        # Link to results (if available)
        link_html = ""
        if has_index:
            link_html = f'<a href="{index_path}" class="card-link">View Results →</a>'
        else:
            link_html = '<span class="card-link disabled">No Results Available</span>'
        
        card_html = f"""
        <div class="experiment-card" data-model="{model_short.lower()}" data-layer="{layer_cutoff}" data-status="{status}">
            <div class="card-header">
                <h3 class="experiment-title">{experiment_name}</h3>
            </div>
            <div class="card-content">
                <div class="config-item">
                    <span class="config-label">Model:</span>
                    <span class="config-value">{model_short}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Dictionary Size:</span>
                    <span class="config-value">{dict_size}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Last Updated:</span>
                    <span class="config-value">{formatted_time}</span>
                </div>
            </div>
            <div class="card-footer">
                {link_html}
            </div>
        </div>
        """
        cards.append(card_html)
    
    return cards


def generate_html_template(cards: List[str], mapping_data: Dict[str, Any]) -> str:
    """Generate the complete HTML page."""
    
    # Get summary statistics
    stats = mapping_data.get("summary_statistics", {})
    total_experiments = stats.get("total_tomls", 0)
    complete_count = stats.get("tomls_complete", 0)
    partial_count = stats.get("tomls_partial", 0)
    
    cards_html = "\n".join(cards)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Results Dashboard</title>
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({{
      jax: ["input/TeX", "output/HTML-CSS"],
      extensions: ["tex2jax.js"],
      "HTML-CSS": {{ preferredFont: "TeX", availableFonts: ["STIX","TeX"] }},
      tex2jax: {{ inlineMath: [ ["$", "$"], ["\\(","\\)"] ], displayMath: [ ["$$","$$"], ["\\[", "\\]"] ], processEscapes: true, ignoreClass: "tex2jax_ignore|dno" }},
      TeX: {{ noUndefined: {{ attributes: {{ mathcolor: "red", mathbackground: "#FFEEEE", mathsize: "90%" }} }} }},
      messageStyle: "none"
    }});
    </script>    
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js"></script>

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8fafc;
            color: #1a202c;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .stats-bar {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        
        .stats-item {{
            text-align: center;
        }}
        
        .stats-number {{
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }}
        
        .stats-label {{
            font-size: 0.9rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .controls {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .controls h2 {{
            margin-bottom: 1rem;
            color: #2d3748;
        }}
        
        .filter-group {{
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .filter-item {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}
        
        .filter-item label {{
            font-size: 0.9rem;
            color: #4a5568;
            font-weight: 500;
        }}
        
        .filter-item input, .filter-item select {{
            padding: 0.5rem;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        
        .experiments-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
        }}
        
        .experiment-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
            border: 1px solid #e2e8f0;
        }}
        
        .experiment-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }}
        
        .experiment-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #2d3748;
            margin: 0;
        }}
        
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-complete {{
            background: #c6f6d5;
            color: #22543d;
        }}
        
        .status-partial {{
            background: #feebc8;
            color: #c05621;
        }}
        
        .status-unknown {{
            background: #e2e8f0;
            color: #4a5568;
        }}
        
        .card-content {{
            margin-bottom: 1rem;
        }}
        
        .config-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            align-items: center;
        }}
        
        .config-label {{
            font-weight: 500;
            color: #4a5568;
            font-size: 0.9rem;
        }}
        
        .config-value {{
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9rem;
            color: #2d3748;
        }}
        
        .card-footer {{
            border-top: 1px solid #e2e8f0;
            padding-top: 1rem;
        }}
        
        .card-link {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9rem;
            font-weight: 500;
            transition: background 0.2s ease;
        }}
        
        .card-link:hover {{
            background: #5a67d8;
        }}
        
        .card-link.disabled {{
            background: #a0aec0;
            cursor: not-allowed;
        }}
        
        .hidden {{
            display: none !important;
        }}
        
        .no-results {{
            text-align: center;
            padding: 3rem;
            color: #718096;
            font-style: italic;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .stats-bar {{
                flex-direction: column;
                text-align: center;
            }}
            
            .filter-group {{
                flex-direction: column;
                align-items: stretch;
            }}
            
            .experiments-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Experiment Results Dashboard</h1>
            <p>Browse and navigate mechanistic interpretability experiments</p>
        </div>
        
        <div class="stats-bar">
            <div class="stats-item">
                <div class="stats-number">{total_experiments}</div>
                <div class="stats-label">Total Experiments</div>
            </div>
        </div>
        

        
        <div class="experiments-grid" id="experiments-grid">
            {cards_html}
        </div>
        
        <div class="no-results hidden" id="no-results">
            <h3>No experiments match your filters</h3>
            <p>Try adjusting your search criteria</p>
        </div>
    </div>
    
    <script>
        // Filter functionality
        const searchInput = document.getElementById('search');
        const modelFilter = document.getElementById('model-filter');
        const statusFilter = document.getElementById('status-filter');
        const layerFilter = document.getElementById('layer-filter');
        const experimentsGrid = document.getElementById('experiments-grid');
        const noResults = document.getElementById('no-results');
        
        function filterExperiments() {{
            const searchTerm = searchInput.value.toLowerCase();
            const modelValue = modelFilter.value.toLowerCase();
            const statusValue = statusFilter.value.toLowerCase();
            const layerValue = layerFilter.value;
            
            const cards = document.querySelectorAll('.experiment-card');
            let visibleCount = 0;
            
            cards.forEach(card => {{
                const cardText = card.textContent.toLowerCase();
                const cardModel = card.dataset.model || '';
                const cardStatus = card.dataset.status || '';
                const cardLayer = card.dataset.layer || '';
                
                const matchesSearch = !searchTerm || cardText.includes(searchTerm);
                const matchesModel = !modelValue || cardModel.includes(modelValue);
                const matchesStatus = !statusValue || cardStatus === statusValue;
                const matchesLayer = !layerValue || cardLayer === layerValue;
                
                if (matchesSearch && matchesModel && matchesStatus && matchesLayer) {{
                    card.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    card.classList.add('hidden');
                }}
            }});
            
            // Show/hide no results message
            if (visibleCount === 0) {{
                experimentsGrid.classList.add('hidden');
                noResults.classList.remove('hidden');
            }} else {{
                experimentsGrid.classList.remove('hidden');
                noResults.classList.add('hidden');
            }}
        }}
        
        // Attach event listeners
        searchInput.addEventListener('input', filterExperiments);
        modelFilter.addEventListener('change', filterExperiments);
        statusFilter.addEventListener('change', filterExperiments);
        layerFilter.addEventListener('change', filterExperiments);
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.key === '/' && !searchInput.matches(':focus')) {{
                e.preventDefault();
                searchInput.focus();
            }}
        }});
        
        console.log('🧠 Experiment Dashboard loaded successfully!');
        console.log('💡 Tip: Press "/" to quickly search experiments');
    </script>
</body>
</html>"""


def main():
    """Main function to generate the master index."""
    parser = argparse.ArgumentParser(description="Generate master index HTML for experiment results")
    parser.add_argument(
        "--mapping-file", 
        type=Path,
        default="toml_to_results_mapping.json",
        help="Path to the TOML to results mapping JSON file"
    )
    parser.add_argument(
        "--output-file",
        type=Path, 
        default="outputs/index.html",
        help="Path for the generated index HTML file"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=".",
        help="Base path for resolving relative file paths"
    )
    
    args = parser.parse_args()
    
    try:
        # Load mapping data
        print(f"Loading mapping data from {args.mapping_file}...")
        mapping_data = load_mapping_data(args.mapping_file)
        
        # Generate experiment cards
        print("Generating experiment cards...")
        cards = generate_experiment_cards(mapping_data, args.base_path)
        
        # Generate HTML
        print("Generating HTML...")
        html_content = generate_html_template(cards, mapping_data)
        
        # Create output directory if needed
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(args.output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Master index generated successfully: {args.output_file}")
        print(f"📊 Processed {len(cards)} experiments")
        
    except Exception as e:
        print(f"❌ Error generating master index: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())