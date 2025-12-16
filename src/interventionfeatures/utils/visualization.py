import html
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

# Config is just used for type hints, accepts any object with dict_size attribute
# Local imports
from ..core.search import ActivationSimSearcher


def escape_html_text(text: str) -> str:
    """Escapes text for safe HTML display."""
    return html.escape(str(text))


class ActivationSimDisplay:
    """
    Generates a rich, interactive HTML report to visualize activation similarity search results.

    This class is initialized with an ActivationSimSearcher instance to gain access to the
    indexed data and model configurations. Before generating a report, the specific query
    vectors must be set using the `set_query` method.
    """

    def __init__(self, searcher: ActivationSimSearcher):
        """
        Initializes the display helper.

        Args:
            searcher: An initialized and indexed ActivationSimSearcher instance.
        """
        self.searcher = searcher
        self.data_handler = searcher.data_handler
        # Assuming the first data handler's tokenizer is representative
        self.tokenizer = searcher.data_handler.tokenizer
        self.device = searcher.device
        self.pad_token_id = self.tokenizer.pad_token_id
        self.d_model = searcher.d_model
        self.scoring_type = searcher.scoring_type

        self.query_vecs_for_display: torch.Tensor | None = None
        self.original_query_norms: list[float] | None = None

    def set_query(self, query_vecs: torch.Tensor):
        """
        Sets the query vectors for the next display report. This must be called
        before display_search_results.

        Args:
            query_vecs (torch.Tensor): The query vectors (shape: K, d_model) used in the search.
        """
        if query_vecs.dim() == 1:
            query_vecs = query_vecs.unsqueeze(0)  # Shape: (1, d_model)

        # Store original norms before any normalization
        self.original_query_norms = torch.norm(query_vecs, p=2, dim=1).cpu().tolist()

        if self.scoring_type == "cosine":
            self.query_vecs_for_display = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
        else:
            self.query_vecs_for_display = query_vecs
        print(f"Display query set for {self.query_vecs_for_display.shape[0]}-token sequences.")

    @torch.no_grad()
    def display_search_results(
        self,
        results: list[dict],
        scores_by_vec: torch.Tensor,
        output_file_path: str = "activation_similarity_report.html",
        explanation: str | None = None,
        optimal_threshold: float | None = None,
        polarity: str = "positive",
    ):
        """
        Generates and saves a rich, interactive HTML report for the search results.
        Ensure `set_query()` has been called with the correct query vectors first.

        Args:
            results (List[Dict]): The search results from ActivationSimSearcher.
            output_file_path (str): The path to save the generated HTML report.
            explanation (Optional[str]): Optional explanation text for the feature.
            optimal_threshold (Optional[float]): Optional optimal threshold from faithfulness testing.
            polarity (str): Whether this is "positive" or "negative" polarity visualization.
        """
        if self.query_vecs_for_display is None:
            raise ValueError(
                "Query vectors not set. Call `set_query(query_vecs)` before displaying results."
            )

        if not results:
            print("No results to display.")
            return

        K, _ = self.query_vecs_for_display.shape
        # Use the original norms stored before any normalization
        query_norms = self.original_query_norms
        js_results_data = []
        global_max_similarity = float("-inf") * torch.ones(K, device=self.device)

        print("Generating visualization data for top results...")
        # Calculate normalized similarities for this result
        # scores_by_vec has shape (top_k, K, SEQ_LEN)
        result_max_similarities = torch.max(scores_by_vec)  # Global max across all results
        for res_idx, res_item in enumerate(tqdm(results, desc="Processing results")):
            sample_idx = res_item["sample_idx"]
            original_matched_token_pos = res_item["token_pos"]

            full_tokens_tensor_cpu = torch.tensor(self.searcher.indexed_sample_tokens[sample_idx])
            # Extract similarities for this result: shape (K, SEQ_LEN)
            sequence_similarities = scores_by_vec[res_idx]

            non_pad_mask_cpu = full_tokens_tensor_cpu != self.pad_token_id

            # Filter similarities to only active (non-padded) tokens
            active_sequence_sims = sequence_similarities[:, : len(non_pad_mask_cpu)]
            active_tokens_ids = full_tokens_tensor_cpu[non_pad_mask_cpu]

            normalized_sequence_sims = active_sequence_sims.clone()
            for i in range(K):
                if result_max_similarities > 0:
                    normalized_sequence_sims[i] = active_sequence_sims[i] / result_max_similarities

            for i in range(K):
                global_max_similarity[i] = max(
                    global_max_similarity[i], torch.max(active_sequence_sims[i]).item()
                )

            # Map from original padded index to the new un-padded index
            original_indices_in_padded_seq = torch.where(non_pad_mask_cpu)[0].tolist()
            pos_map = {
                orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices_in_padded_seq)
            }
            max_inds = res_item["max_indices"]

            tokens_str_list_escaped = [
                escape_html_text(self.tokenizer.decode([t_id])) for t_id in active_tokens_ids
            ]

            max_token_ids = [active_tokens_ids[i] for i in max_inds]
            matched_tokens_str = escape_html_text(self.tokenizer.decode(max_token_ids))

            full_text_reconstructed = escape_html_text(
                self.tokenizer.decode(active_tokens_ids, skip_special_tokens=True)
            )

            js_results_data.append(
                {
                    "id": f"result_{res_idx}",
                    "sample_idx": sample_idx,
                    "score": res_item["score"],
                    "highlight_inds": max_inds,
                    "matched_tokens_str": matched_tokens_str,
                    "original_pos": original_matched_token_pos,
                    "full_sequence_tokens": tokens_str_list_escaped,
                    "sequence_similarities": active_sequence_sims.transpose(0, 1).cpu().tolist(),
                    "normalized_similarities": normalized_sequence_sims.transpose(0, 1)
                    .cpu()
                    .tolist(),
                    "result_max_similarities": result_max_similarities.cpu().tolist(),
                    "full_text_display": full_text_reconstructed,
                    "K": K,
                }
            )

        global_max_similarity = global_max_similarity.cpu().tolist()

        # Generate and save the HTML file
        self._write_html_report(
            js_results_data,
            K,
            global_max_similarity,
            output_file_path,
            explanation,
            query_norms,
            optimal_threshold,
            polarity,
        )

    def _write_html_report(
        self,
        js_results_data,
        K,
        global_max_similarity,
        output_file_path,
        explanation: str | None = None,
        query_norms: list[float] | None = None,
        optimal_threshold: float | None = None,
        polarity: str = "positive",
    ):
        """Generates the final HTML content and writes it to a file."""
        polarity_title = f"({polarity.capitalize()})" if polarity != "positive" else ""
        polarity_badge = (
            f'<span style="background-color: {"#28a745" if polarity == "positive" else "#dc3545"}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin-left: 10px;">{polarity.upper()}</span>'
            if polarity
            else ""
        )

        html_content = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Token Activation Similarity Report {polarity_title}</title><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; color: #212529; }}
.container {{ background-color: #ffffff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
h1 {{ color: #0056b3; text-align: center; margin-bottom: 25px; font-weight: 300;}}
h2 {{ color: #0056b3; border-bottom: 1px solid #dee2e6; padding-bottom: 10px; margin-top:0; font-size: 1.5em; font-weight: 400; }}
.token {{ padding: 3px 5px; margin: 2px 1px; border-radius: 4px; display: inline-block; border: 1px solid #ced4da; cursor: default; transition: transform 0.1s ease-out, box-shadow 0.1s ease-out; font-size: 0.9em; line-height: 1.5; }}
.token:hover {{ transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.token.highlight-window {{ border: 2px solid #c82333; font-weight: 600; color: #721c24; }}
.sequence-display {{ margin-top: 12px; padding: 12px; border: 1px solid #e9ecef; background-color: #f8f9fa; border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.7; }}
.full-text-display {{ margin-top: 15px; padding: 10px; border: 1px solid #dfe6e9; background-color: #f9f9f9; border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.6; font-size: 0.95em; color: #333; }}
.tooltip {{ position: absolute; background-color: rgba(33,37,41,0.9); color: white; padding: 8px 12px; border-radius: 5px; visibility: hidden; z-index: 1000; pointer-events: none; font-size: 0.85em; max-width: 400px; }}
.details p {{ margin: 5px 0; font-size: 0.9em; color: #495057; }} .details strong {{ color: #0056b3; }}
.footer-note {{ font-size: 0.8em; color: #6c757d; text-align: center; margin-top: 30px; }}
.global-max-info {{ background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-size: 0.9em; color: #1565c0; }}
.threshold-control {{ margin-top: 15px; padding: 10px; background-color: rgba(255,255,255,0.3); border-radius: 6px; }}
.threshold-slider {{ width: 200px; margin: 0 10px; }}
.threshold-value {{ font-weight: bold; color: #0056b3; }}
</style></head><body>
{'<div class="explanation-section" style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #0056b3;"><h3 style="color: #0056b3; margin-top: 0;">Feature Explanation</h3><div style="line-height: 1.6; color: #333;">' + escape_html_text(explanation) + "</div></div>" if explanation else ""}<div class="global-max-info">
    <strong>Query Type:</strong> {K}-token sequences {polarity_badge}<br>
    {"<strong>Query Vector Norms:</strong> " + str([f"{norm:.4f}" for norm in query_norms]) + "<br>" if query_norms else ""}
    <strong>Global Max Sequence Similarities:</strong> {global_max_similarity}<br>
    <div style="margin-top: 10px;">
        <label style="font-weight: normal; cursor: pointer;">
            <input type="checkbox" id="normalizeToggle" checked style="margin-right: 8px;">
            Show Normalized Activations (divide by max per result)
        </label>
        <span id="normalizationStatus" style="margin-left: 15px; font-style: italic; color: #666;"></span>
    </div>
    <div class="threshold-control">
        <label style="font-weight: normal;">
            Activation Cutoff Threshold: <span class="threshold-value" id="thresholdValue">{optimal_threshold if optimal_threshold is not None else 0.0:.3f}</span>
        </label><br>
        <input type="range" id="thresholdSlider" class="threshold-slider" 
               min="0" max="1" step="0.001" 
               value="{optimal_threshold if optimal_threshold is not None else 0.0}">
        <div style="margin-top: 5px; font-size: 0.8em; color: #666;">
            {f"Default from faithfulness testing: {optimal_threshold:.3f}" if optimal_threshold is not None else "No faithfulness data available - adjust manually"}
        </div>
    </div>
</div>
<div id="tooltip" class="tooltip"></div>
"""
        for res_data in js_results_data:
            html_content += f"""
<div class="container" id="container_{res_data["id"]}">
    <h2>Result {int(res_data["id"].split("_")[1]) + 1} (Sample Ref: {res_data["sample_idx"]})</h2>
    <div class="details">
        <p><strong>FAISS Score (Similarity):</strong> {res_data["score"]:.4f}</p>
        <p><strong>Matched {K}-Token Window:</strong> <span style="background-color: #f8d7da; padding: 2px; border-radius:3px;">"{res_data["matched_tokens_str"]}"</span></p>
        <p><strong>Window Start Position:</strong> {res_data["original_pos"]}</p>
    </div>
    <div class="sequence-display" id="sequence_display_{res_data["id"]}"></div>
    <div class="full-text-display" id="full_text_display_{res_data["id"]}"></div>
</div>"""

        html_content += f"""
<script>
    const allResultsData = {json.dumps(js_results_data)};
    const globalMaxSequenceSimilarity = {global_max_similarity};
    const polarity = "{polarity}";
    const tooltip = document.getElementById('tooltip');
    let isNormalized = true;
    
    // Parse threshold from URL hash parameter, fallback to optimal_threshold or 0.0
    function getThresholdFromUrl() {{
        const hash = window.location.hash;
        if (hash && hash.includes('threshold=')) {{
            const match = hash.match(/threshold=([\\d\\.]+)/);
            if (match && match[1]) {{
                const urlThreshold = parseFloat(match[1]);
                if (!isNaN(urlThreshold)) {{
                    return urlThreshold;
                }}
            }}
        }}
        return {optimal_threshold if optimal_threshold is not None else 0.0};
    }}
    
    let currentThreshold = getThresholdFromUrl();

    function htmlDecode(input) {{
        const doc = new DOMParser().parseFromString(input, "text/html");
        return doc.documentElement.textContent;
    }}

    function getColorForSimilarity(sim, maxSim, isNormalizedMode = false, threshold = 0.0) {{
        if (sim === -Infinity || maxSim <= 0) {{
            return {{ backgroundColor: '#f8f9fa', color: '#6c757d' }};
        }}

        // Apply threshold cutoff - if below threshold, return transparent/no color
        if (sim <= threshold) {{
            return {{ backgroundColor: 'transparent', color: '#6c757d' }};
        }}

        let t;
        if (isNormalizedMode) {{
            // In normalized mode, use a more sensitive scale since values are 0-1
            // Apply a power function to make smaller differences more visible
            const normalizedT = Math.max(0, Math.min(1, sim / maxSim));
            t = Math.pow(normalizedT, 0.7); // Makes mid-range values more visible
        }} else {{
            // In raw mode, use linear scaling
            t = Math.max(0, Math.min(1, sim / maxSim));
        }}

        let r, g, b;
        if (polarity === "negative") {{
            // Use white to blue gradient for negative polarity
            b = 255;
            r = g = Math.round(255 * (1 - t));
        }} else {{
            // Use white to red gradient for positive polarity
            r = 255;
            g = b = Math.round(255 * (1 - t));
        }}

        const textColor = (r + g + b) / 3 > 127 ? '#000000' : '#FFFFFF';
        return {{ backgroundColor: `rgb(${{r}},${{g}},${{b}})`, color: textColor }};
    }}

    function renderResult(resultData) {{
        const displayDiv = document.getElementById(`sequence_display_${{resultData.id}}`);
        const fullTextDiv = document.getElementById(`full_text_display_${{resultData.id}}`);
        if (!displayDiv) return;

        // Choose which similarities to use based on toggle state
        const currentSimilarities = isNormalized ? resultData.normalized_similarities : resultData.sequence_similarities;
        const currentMaxSimilarities = isNormalized ? Array(resultData.K).fill(1) : globalMaxSequenceSimilarity; // Normalized max is always 1

        displayDiv.innerHTML = '';
        resultData.full_sequence_tokens.forEach((tokenHtml, i) => {{
            const span = document.createElement('span');
            span.className = 'token';
            span.innerHTML = tokenHtml;

            const seqSim = currentSimilarities[i];
            const rawSeqSim = resultData.sequence_similarities[i];
            let maxIdx = 0
            let K = resultData.K;
            let maxSim = Number.NEGATIVE_INFINITY;
            for (let j = 0; j < K; j++) {{
                if (currentSimilarities[i] && currentSimilarities[i][j] > maxSim) {{
                    maxIdx = j;
                    maxSim = currentSimilarities[i][j];
                }}
            }}

            try{{
                const colors = getColorForSimilarity(seqSim[maxIdx], currentMaxSimilarities[maxIdx], isNormalized, currentThreshold);
                span.style.backgroundColor = colors.backgroundColor;
                span.style.color = colors.color;
            }} catch {{ console.log("Get color for sim doesn't work") }}

            if (resultData.highlight_inds.includes(i)) {{
                span.classList.add('highlight-window');
            }}

            span.onmousemove = e => {{
                tooltip.style.visibility = 'visible';
                let rawTokenSimsText = resultData.sequence_similarities[i].map((s, j) => `Q${{j+1}}: ${{s.toFixed(4)}}`).join(', ');
                let normalizedTokenSimsText = resultData.normalized_similarities[i].map((s, j) => `Q${{j+1}}: ${{s.toFixed(4)}}`).join(', ');

                let tooltipContent = `<b>${{htmlDecode(tokenHtml)}}</b> (Index: ${{i}})<br>`;
                if (isNormalized) {{
                    tooltipContent += `Normalized Sim: ${{seqSim[maxIdx].toFixed(4)}}<br>`;
                    tooltipContent += `Raw Sim: ${{rawSeqSim[maxIdx].toFixed(4)}}<br>`;
                    tooltipContent += `Normalized Token Sims: ${{normalizedTokenSimsText}}<br>`;
                }} else {{
                    tooltipContent += `Raw Sim: ${{seqSim[maxIdx].toFixed(4)}}<br>`;
                    tooltipContent += `Raw Token Sims: ${{rawTokenSimsText}}<br>`;
                }}
                tooltipContent += `Vector index match: ${{maxIdx + 1}}`;

                tooltip.innerHTML = tooltipContent;
                tooltip.style.left = `${{e.pageX + 15}}px`;
                tooltip.style.top = `${{e.pageY + 15}}px`;
            }};
            span.onmouseout = () => {{ tooltip.style.visibility = 'hidden'; }};
            displayDiv.appendChild(span);
        }});

        if (fullTextDiv && resultData.full_text_display) {{
            fullTextDiv.innerHTML = `<p><strong>Full Text:</strong></p><p>${{resultData.full_text_display}}</p>`;
        }}
    }}

    function toggleNormalization() {{
        isNormalized = !isNormalized;
        const statusSpan = document.getElementById('normalizationStatus');
        const colorGradient = polarity === "negative" ? "linear-gradient(90deg, white, lightblue, blue)" : "linear-gradient(90deg, white, pink, red)";
        const colorLabel = polarity === "negative" ? "White→Blue colors" : "White→Red colors";
        
        if (isNormalized) {{
            statusSpan.innerHTML = `Showing normalized values (max = 1.0 per result) | <span style="background: ${{colorGradient}}; padding: 2px 8px; border-radius: 3px; color: black; font-size: 0.8em;">${{colorLabel}}</span>`;
            statusSpan.style.color = '#28a745';
        }} else {{
            statusSpan.innerHTML = `Showing raw values (global scale) | <span style="background: ${{colorGradient}}; padding: 2px 8px; border-radius: 3px; color: black; font-size: 0.8em;">${{colorLabel}}</span>`;
            statusSpan.style.color = '#666';
        }}

        // Re-render all results with new normalization setting
        allResultsData.forEach(renderResult);
    }}

    function updateThreshold() {{
        const slider = document.getElementById('thresholdSlider');
        const valueDisplay = document.getElementById('thresholdValue');
        currentThreshold = parseFloat(slider.value);
        valueDisplay.textContent = currentThreshold.toFixed(3);
        
        // Re-render all results with new threshold setting
        allResultsData.forEach(renderResult);
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        // Initialize normalization status (start with normalized mode)
        const statusSpan = document.getElementById('normalizationStatus');
        const colorGradient = polarity === "negative" ? "linear-gradient(90deg, white, lightblue, blue)" : "linear-gradient(90deg, white, pink, red)";
        const colorLabel = polarity === "negative" ? "White→Blue colors" : "White→Red colors";
        statusSpan.innerHTML = `Showing normalized values (max = 1.0 per result) | <span style="background: ${{colorGradient}}; padding: 2px 8px; border-radius: 3px; color: black; font-size: 0.8em;">${{colorLabel}}</span>`;
        statusSpan.style.color = '#28a745';

        // Set up toggle event listener
        const toggleCheckbox = document.getElementById('normalizeToggle');
        toggleCheckbox.addEventListener('change', toggleNormalization);

        // Set up threshold slider event listener and initialize with URL parameter value
        const thresholdSlider = document.getElementById('thresholdSlider');
        const thresholdValue = document.getElementById('thresholdValue');
        thresholdSlider.addEventListener('input', updateThreshold);
        
        // Set slider value from URL parameter or default
        thresholdSlider.value = currentThreshold;
        thresholdValue.textContent = currentThreshold.toFixed(3);

        // Render initial results
        allResultsData.forEach(renderResult);
    }});
</script>
</body></html>"""
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"\nSuccess: HTML report saved to {output_file_path}")

            # Also save JSON for Streamlit app
            json_path = output_file_path.replace(".html", ".json")
            json_data = {
                "results": js_results_data,
                "global_max_similarity": global_max_similarity,
                "K": K,
                "polarity": polarity,
                "explanation": explanation,
                "query_norms": query_norms,
                "optimal_threshold": optimal_threshold,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f)
            print(f"Success: JSON data saved to {json_path}")
        except OSError as e:
            print(f"\nError: Failed to write HTML report to file: {e}")


def extract_faithfulness_data(output_path: str) -> dict[int, float]:
    """
    Extract faithfulness scores from validation results if available.

    Args:
        output_path: Path to output directory containing validation results

    Returns:
        Dictionary mapping direction indices to their faithfulness scores.
    """

    output_dir = Path(output_path)
    faithfulness_scores = {}

    # Try to load compressed validation results first
    # Try regular validation results
    results_file = output_dir / "validation_results.json"
    if not results_file.exists():
        return faithfulness_scores

    try:
        with open(results_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return faithfulness_scores

    def scores_from_direction(data):
        faithfulness_scores = {}
        # Extract scores from aggregated results
        validation_results = data.get("validation_results", {})
        aggregated_results = validation_results.get("aggregated_results", {})
        direction_summaries = aggregated_results.get("source_runs")[0].get("direction_summaries")
        aggregated_directions = aggregated_results.get("aggregated_directions", [])

        for i, summary in enumerate(direction_summaries):
            accuracy = summary.get("mean_accuracy", -1)
            optimal_threshold = direction_summaries[i].get("optimal_threshold")
            direction_idx = summary["direction_idx"]
            faithfulness_scores[direction_idx] = (accuracy, optimal_threshold)

        return faithfulness_scores

    return scores_from_direction(data)


def find_related_direction_files(output_path: str) -> list[str]:
    """
    Find direction visualization files within the given output path.

    This function looks for direction visualization files in the html
    subdirectory of the provided output_path.

    Args:
        output_path: Current output directory path.

    Returns:
        List of local file paths to direction visualization files.
    """
    from pathlib import Path

    output_dir = Path(output_path)
    html_dir = output_dir / "html"
    direction_files = []

    if html_dir.is_dir():
        for file_path in html_dir.glob("direction_*_visualization.html"):
            direction_files.append(f"html/{file_path.name}")
        # Also include older format files with positive/negative suffix for backward compatibility
        for file_path in html_dir.glob("direction_*_positive_visualization.html"):
            direction_files.append(f"html/{file_path.name}")
        for file_path in html_dir.glob("direction_*_negative_visualization.html"):
            direction_files.append(f"html/{file_path.name}")

    return sorted(direction_files)


def generate_index_page(
    output_path: str,
    config_dict: dict,
    config: Any,
    metadata: dict | None = None,
) -> str:
    """
    Generate an index.html page for a run with navigation to all HTML visualizations.

    Args:
        output_path: Directory where index.html will be saved
        config_dict: Configuration dictionary with run parameters
        html_files: List of HTML file paths relative to output_path
        metadata: Optional metadata dictionary with additional info

    Returns:
        Path to the generated index.html file
    """
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(output_path)
    index_file = output_dir / "index.html"
    html_dir = output_dir / "html"

    # Extract metadata
    timestamp = (
        metadata.get("timestamp", datetime.now().isoformat())
        if metadata
        else datetime.now().isoformat()
    )
    model_name = config_dict.get("model_name", "Unknown")
    dataset_name = config_dict.get("dataset_name", "Unknown")
    dict_size = config_dict.get("dict_size", "Unknown")
    layer_cutoff = config_dict.get("layer_cutoff", "Unknown")
    target_token_offset = config_dict.get("target_token_offset", "Unknown")

    direction_idx_to_file = [
        f"{i // 2}_{'positive' if i % 2 == 0 else 'negative'}_visualization.html"
        for i in range(config.dict_size)
    ]

    # Extract faithfulness scores and threshold data for direction sorting and URL parameters
    faithfulness_data = extract_faithfulness_data(output_path)

    # Extract direction numbers from files and sort by faithfulness score
    # direction_nums = []
    # for f in direction_files:
    #    file_name = Path(f).name
    #    parts = file_name.split('_')
    #    if len(parts) >= 2 and parts[0].isdigit():
    #        direction_nums.append(int(parts[0]))

    # print("SORTED", direction_nums)
    ## Sort by faithfulness score (use positive scores as default)
    # sorted_direction_nums = sorted(set(direction_nums), key=lambda num: faithfulness_scores_pos.get(num, -1.0) if faithfulness_scores_pos else 0, reverse=True)
    # print("SORTED", sorted_direction_nums)

    # Load explanations if available
    explanations_map = {}

    explanations_file = output_dir / "explanations.json"
    if explanations_file.exists():
        try:
            import json

            with open(explanations_file, encoding="utf-8") as f:
                data = json.load(f)
                if "explanations" in data:
                    for explanation in data["explanations"]:
                        direction_idx = explanation.get("direction_index")
                        explanation_text = explanation.get("explanation", "")
                        if direction_idx is not None and explanation_text:
                            explanations_map[direction_idx] = explanation_text
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load explanations from {explanations_file}: {e}")

    # faithfulness_files = [or f in html_files if "faithfulness_" in f]
    # summary_files = [f for f in html_files if "summary" in f or "aggregated" in f] # TODO: put in

    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mechanistic Interpretability Results - {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f8fafc;
            color: #1a202c;
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
            margin: 0 0 0.5rem 0;
            font-size: 2.5rem;
            font-weight: 600;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        .metadata {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .metadata h2 {{
            margin: 0 0 1rem 0;
            color: #2d3748;
            font-size: 1.3rem;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        .metadata-item {{
            padding: 0.75rem;
            background: #f7fafc;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        .metadata-label {{
            font-weight: 600;
            color: #4a5568;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }}
        .metadata-value {{
            color: #2d3748;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.95rem;
        }}
        .section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            margin: 0 0 1rem 0;
            color: #2d3748;
            font-size: 1.3rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }}
        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .file-card {{
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            transition: all 0.2s ease;
            background: #fafafa;
        }}
        .file-card:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .file-card a {{
            text-decoration: none;
            color: #2d3748;
            display: block;
        }}
        .file-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #4a5568;
        }}
        .file-description {{
            font-size: 0.9rem;
            color: #718096;
            line-height: 1.4;
        }}
        .explanation-preview {{
            font-size: 0.85rem;
            color: #a0aec0;
            font-style: italic;
            margin-top: 0.5rem;
            line-height: 1.3;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: #667eea;
            color: white;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }}
        .empty-section {{
            text-align: center;
            color: #a0aec0;
            font-style: italic;
            padding: 2rem;
        }}
        .footer {{
            text-align: center;
            color: #718096;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e2e8f0;
        }}

        .positive {{
            background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin-left: 10px;
        }}
        .negative {{
            background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="metadata">
        <h2>📋 Run Information</h2>
        <div class="metadata-grid">
            <div class="metadata-item">
                <div class="metadata-label">Model</div>
                <div class="metadata-value">{model_name}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Dataset</div>
                <div class="metadata-value">{dataset_name}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Dictionary Size</div>
                <div class="metadata-value">{dict_size}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Layer</div>
                <div class="metadata-value">{layer_cutoff}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Target Token Offset</div>
                <div class="metadata-value">{target_token_offset}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Generated</div>
                <div class="metadata-value">{timestamp[:19].replace("T", " ")}</div>
            </div>
        </div>
    </div>"""

    # Direction visualizations section
    if True:
        sorting_note = ""
        # if faithfulness_data:
        #    sorting_note = '<div style="text-align: center; margin-bottom: 15px; color: #718096; font-style: italic;">Directions sorted by faithfulness score (highest first)</div>'

        html_content += f"""
    <div class="section">
        <h2>🎯 Feature Direction Visualizations</h2>
        {sorting_note}
        <div class="file-grid">"""

        for idx, direction_file in enumerate(direction_idx_to_file):
            # Get faithfulness score and explanation for this direction
            faithfulness_info = ""
            explanation_info = ""
            optimal_threshold = 0
            try:
                if faithfulness_data:
                    score, _optimal_threshold = faithfulness_data[idx]
                    optimal_threshold = _optimal_threshold

                    faithfulness_info = f" • Faithfulness: {score:.0%}"

                if idx in explanations_map:
                    explanation_text = explanations_map[idx]
                    max_chars = 150
                    if len(explanation_text) > max_chars:
                        truncated = explanation_text[:max_chars].rsplit(" ", 1)[0]
                        explanation_info = f'<div class="explanation-preview">{truncated}...</div>'
                    else:
                        explanation_info = (
                            f'<div class="explanation-preview">{explanation_text}</div>'
                        )
            except (ValueError, TypeError):
                pass

            if direction_file:
                title_prefix = ""
                file_name = Path(direction_file).name
                base_href = (
                    direction_file if direction_file.startswith("html/") else f"html/{file_name}"
                )
                base_href += "#threshold=" + str(optimal_threshold)

                html_content += f"""
                <div class="file-card">
                    <a href="{base_href}">
                        <div class="file-title">{title_prefix}Direction {idx // 2} <span {
                    "class='positive'> Positive" if idx % 2 == 0 else "class='negative'> Negative"
                }
                                                                                          </span> </div>
                        <div class="file-description">
                            Interactive visualization showing feature activation patterns
                            {faithfulness_info}
                            {explanation_info}
                        </div>
                        <span class="badge">Direction</span>
                    </a>
                </div>"""

        html_content += """
        </div>
    </div>"""
    else:
        html_content += """
    <div class="section">
        <h2>🎯 Feature Direction Visualizations</h2>
        <div class="empty-section">No direction visualizations found.</div>
    </div>"""

    # Summary/aggregated results section
    if False and summary_files:
        html_content += """
    <div class="section">
        <h2>📊 Summary Reports</h2>
        <div class="file-grid">"""

        for file_path in sorted(summary_files):
            file_name = Path(file_path).name
            if "summary" in file_name:
                title = "Faithfulness Summary"
                description = (
                    "Comprehensive summary of all faithfulness testing results across directions."
                )
            elif "aggregated" in file_name:
                title = "Aggregated Analysis"
                description = "Statistical aggregation of multiple validation trials with confidence intervals."
            else:
                title = file_name.replace("_", " ").title()
                description = "Summary analysis results."

            html_content += f"""
            <div class="file-card">
                <a href="html/{file_name}">
                    <div class="file-title">{title}</div>
                    <div class="file-description">{description}</div>
                    <span class="badge">Summary</span>
                </a>
            </div>"""

        html_content += """
        </div>
    </div>"""

    # Faithfulness testing section
    if False and faithfulness_files:
        # Group faithfulness files by direction
        faithfulness_by_direction = {}
        for file_path in faithfulness_files:
            file_name = Path(file_path).name
            if "direction_" in file_name and "trial_" in file_name:
                # Extract direction number
                parts = file_name.split("_")
                direction_idx = None
                for i, part in enumerate(parts):
                    if part == "direction" and i + 1 < len(parts):
                        direction_idx = parts[i + 1]
                        break

                if direction_idx:
                    if direction_idx not in faithfulness_by_direction:
                        faithfulness_by_direction[direction_idx] = []
                    faithfulness_by_direction[direction_idx].append(file_path)

        if faithfulness_by_direction:
            html_content += """
    <div class="section">
        <h2>🔍 Faithfulness Testing</h2>
        <div class="file-grid">"""

            for direction_idx in sorted(faithfulness_by_direction.keys()):
                files = faithfulness_by_direction[direction_idx]
                trial_count = len(files)

                html_content += f"""
            <div class="file-card">
                <div class="file-title">Direction {direction_idx} Faithfulness</div>
                <div class="file-description">
                    {trial_count} trial{"s" if trial_count != 1 else ""} testing how well explanations 
                    predict feature activation through targeted interventions.
                </div>
                <div style="margin-top: 0.5rem;">"""

                for file_path in sorted(files):
                    file_name = Path(file_path).name
                    trial_num = (
                        file_name.split("_")[-1].replace(".html", "")
                        if "trial_" in file_name
                        else "0"
                    )
                    html_content += f"""
                    <a href="html/{file_name}" style="display: inline-block; margin: 0.25rem 0.5rem 0.25rem 0; padding: 0.25rem 0.5rem; background: #edf2f7; border-radius: 4px; font-size: 0.8rem; text-decoration: none; color: #4a5568;">Trial {trial_num}</a>"""

                html_content += """
                </div>
                <span class="badge">Validation</span>
            </div>"""

            html_content += """
        </div>
    </div>"""

    # Footer
    html_content += """
    <div class="footer">
    </div>
</body>
</html>"""

    # Write the index file
    try:
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✓ Index page generated: {index_file}")
        return str(index_file)
    except OSError as e:
        print(f"Error: Failed to write index page: {e}")
        return ""
