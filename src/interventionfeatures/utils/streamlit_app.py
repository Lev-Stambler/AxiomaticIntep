#!/usr/bin/env python3
"""Streamlit visualization app for intervention features."""

import argparse
import html
import json
from pathlib import Path

import streamlit as st
from omegaconf import OmegaConf


def get_color_for_similarity(
    sim: float,
    max_sim: float,
    polarity: str,
    threshold: float,
    normalize: bool = True,
) -> tuple[str, str]:
    """
    Calculate background and text color for a similarity value.

    Args:
        sim: The similarity value
        max_sim: Maximum similarity for normalization
        polarity: "positive" or "negative"
        threshold: Cutoff threshold below which to show transparent
        normalize: Whether to apply normalization

    Returns:
        Tuple of (background_color, text_color)
    """
    if sim == float("-inf") or max_sim <= 0:
        return "#f8f9fa", "#6c757d"

    if sim <= threshold:
        return "transparent", "#6c757d"

    if normalize:
        t = max(0, min(1, sim / max_sim))
        t = t**0.7  # Power function for visibility
    else:
        t = max(0, min(1, sim / max_sim))

    if polarity == "negative":
        # White to blue gradient
        b = 255
        r = g = int(255 * (1 - t))
    else:
        # White to red gradient
        r = 255
        g = b = int(255 * (1 - t))

    text_color = "#000000" if (r + g + b) / 3 > 127 else "#FFFFFF"
    return f"rgb({r},{g},{b})", text_color


def render_token_heatmap(
    tokens: list[str],
    similarities: list[list[float]],
    highlight_indices: list[int],
    polarity: str,
    threshold: float,
    normalize: bool,
    global_max: list[float],
) -> None:
    """Render token sequence with color-coded activation heatmap."""
    css = """
    <style>
    .token-container { line-height: 2.0; }
    .token {
        padding: 3px 5px; margin: 2px 1px; border-radius: 4px;
        display: inline-block; border: 1px solid #ced4da;
        cursor: default; font-size: 0.9em;
    }
    .token.highlight-window {
        border: 2px solid #c82333; font-weight: 600;
    }
    </style>
    """

    html_parts = ['<div class="token-container">']

    for i, token in enumerate(tokens):
        sim_values = similarities[i] if i < len(similarities) else [0]
        max_sim = max(sim_values) if sim_values else 0
        max_idx = sim_values.index(max_sim) if sim_values else 0

        ref_max = 1.0 if normalize else (global_max[max_idx] if max_idx < len(global_max) else 1.0)
        bg_color, text_color = get_color_for_similarity(
            max_sim, ref_max, polarity, threshold, normalize
        )

        highlight_class = "highlight-window" if i in highlight_indices else ""
        escaped_token = html.escape(token)

        tooltip = f"Token: {escaped_token} | Sim: {max_sim:.4f} | Vec: {max_idx + 1}"

        html_parts.append(
            f'<span class="token {highlight_class}" '
            f'style="background-color: {bg_color}; color: {text_color};" '
            f'title="{tooltip}">{escaped_token}</span>'
        )

    html_parts.append("</div>")
    st.markdown(css + "".join(html_parts), unsafe_allow_html=True)


def load_results_directory(path: Path) -> dict:
    """Load all result files from an output directory."""
    results = {}

    # Load config YAML
    config_path = path / "config.yaml"
    if config_path.exists():
        results["config"] = OmegaConf.to_container(OmegaConf.load(config_path))

    # Load explanations JSON
    explanations_path = path / "explanations.json"
    if explanations_path.exists():
        with open(explanations_path) as f:
            data = json.load(f)
            results["explanations"] = {}
            for exp in data.get("explanations", []):
                idx = exp.get("direction_index")
                if idx is not None:
                    results["explanations"][idx] = exp.get("explanation", "")

    # Load validation results JSON
    validation_path = path / "validation_results.json"
    if validation_path.exists():
        with open(validation_path) as f:
            results["validation"] = json.load(f)

    # Load direction JSON files
    html_dir = path / "html"
    results["directions"] = {}
    if html_dir.exists():
        for json_file in html_dir.glob("*_visualization.json"):
            with open(json_file) as f:
                data = json.load(f)
                # Extract direction info from filename
                name = json_file.stem.replace("_visualization", "")
                results["directions"][name] = data

    return results


def extract_faithfulness_data(results: dict) -> dict[int, tuple[float, float]]:
    """Extract faithfulness scores and thresholds from validation results."""
    faithfulness_data = {}

    validation = results.get("validation", {})
    validation_results = validation.get("validation_results", {})
    aggregated = validation_results.get("aggregated_results", {})
    source_runs = aggregated.get("source_runs", [])

    if source_runs:
        summaries = source_runs[0].get("direction_summaries", [])
        for summary in summaries:
            idx = summary.get("direction_idx")
            accuracy = summary.get("mean_accuracy", -1)
            threshold = summary.get("optimal_threshold", 0.0)
            if idx is not None:
                faithfulness_data[idx] = (accuracy, threshold)

    return faithfulness_data


def render_dashboard(results: dict, output_path: Path) -> str | None:
    """Render the main dashboard view. Returns selected direction key if clicked."""
    st.title("Intervention Features Visualization")

    # Metadata section
    config = results.get("config", {})
    with st.expander("Run Information", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Model", config.get("model", {}).get("model_name", "Unknown"))
        cols[1].metric("Layer", config.get("model", {}).get("layer_cutoff", "?"))
        cols[2].metric("Dict Size", config.get("training", {}).get("dict_size", "?"))
        cols[3].metric("Target Offset", config.get("model", {}).get("target_token_offset", "?"))

    # Extract faithfulness data
    faithfulness_data = extract_faithfulness_data(results)
    explanations = results.get("explanations", {})
    directions = results.get("directions", {})

    if not directions:
        st.warning(
            "No direction visualization data found. Run the pipeline with visualization enabled first."
        )
        return None

    st.subheader("Feature Directions")

    # Sort directions by name
    sorted_keys = sorted(
        directions.keys(), key=lambda x: (int(x.split("_")[0]), x.split("_")[1] != "positive")
    )

    # Create grid
    cols = st.columns(3)
    for i, key in enumerate(sorted_keys):
        parts = key.split("_")
        dir_idx = int(parts[0])
        polarity = parts[1] if len(parts) > 1 else "positive"

        with cols[i % 3]:
            # Card container
            with st.container():
                # Header with polarity badge
                badge_color = "#28a745" if polarity == "positive" else "#dc3545"
                st.markdown(
                    f"**Direction {dir_idx}** "
                    f'<span style="background-color: {badge_color}; color: white; '
                    f'padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">'
                    f"{polarity.upper()}</span>",
                    unsafe_allow_html=True,
                )

                # Faithfulness score
                if dir_idx in faithfulness_data:
                    accuracy, _ = faithfulness_data[dir_idx]
                    st.caption(f"Faithfulness: {accuracy:.0%}")

                # Explanation preview
                if dir_idx in explanations:
                    exp_text = explanations[dir_idx]
                    if len(exp_text) > 100:
                        exp_text = exp_text[:100] + "..."
                    st.caption(f"_{exp_text}_")

                # View button
                if st.button("View Details", key=f"view_{key}"):
                    return key

    return None


def render_direction_detail(key: str, results: dict) -> bool:
    """Render detailed view for a direction. Returns True if back button clicked."""
    if st.button("Back to Dashboard"):
        return True

    data = results["directions"][key]
    parts = key.split("_")
    dir_idx = int(parts[0])
    polarity = data.get("polarity", parts[1] if len(parts) > 1 else "positive")

    # Header
    badge_color = "#28a745" if polarity == "positive" else "#dc3545"
    st.markdown(
        f"# Direction {dir_idx} "
        f'<span style="background-color: {badge_color}; color: white; '
        f'padding: 4px 12px; border-radius: 6px;">{polarity.upper()}</span>',
        unsafe_allow_html=True,
    )

    # Explanation
    explanation = data.get("explanation")
    if explanation:
        st.info(explanation)

    # Query info
    K = data.get("K", 1)
    global_max = data.get("global_max_similarity", [1.0] * K)
    query_norms = data.get("query_norms")

    norms_info = f" | Query norms: {query_norms}" if query_norms else ""
    st.caption(
        f"Query type: {K}-token sequences | Global max similarities: {global_max}{norms_info}"
    )

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        normalize = st.checkbox("Normalize activations", value=True, key=f"norm_{key}")
    with col2:
        default_threshold = data.get("optimal_threshold", 0.0) or 0.0
        threshold = st.slider(
            "Activation threshold",
            0.0,
            1.0,
            float(default_threshold),
            step=0.01,
            key=f"thresh_{key}",
        )

    # Search results
    st.subheader("Top Activating Sequences")
    search_results = data.get("results", [])

    for i, result in enumerate(search_results):
        score = result.get("score", 0)
        matched_tokens = result.get("matched_tokens_str", "")

        with st.expander(
            f'Result {i + 1} (Score: {score:.4f}) - "{matched_tokens}"', expanded=i < 3
        ):
            # Token heatmap
            tokens = result.get("full_sequence_tokens", [])
            sims = result.get(
                "normalized_similarities" if normalize else "sequence_similarities", []
            )
            highlight_inds = result.get("highlight_inds", [])

            render_token_heatmap(
                tokens=tokens,
                similarities=sims,
                highlight_indices=highlight_inds,
                polarity=polarity,
                threshold=threshold,
                normalize=normalize,
                global_max=global_max,
            )

            # Full text
            full_text = result.get("full_text_display", "")
            if full_text:
                st.text_area(
                    "Full Text", full_text, height=100, disabled=True, key=f"text_{key}_{i}"
                )

    return False


def main():
    """Main entry point for Streamlit app."""
    st.set_page_config(
        page_title="Intervention Features Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Parse command line args
    parser = argparse.ArgumentParser(
        description="Streamlit visualization for intervention features"
    )
    parser.add_argument("output_dir", nargs="?", help="Path to output directory")
    args, _ = parser.parse_known_args()

    # Sidebar for directory selection
    with st.sidebar:
        st.header("Settings")

        # Directory input
        default_dir = args.output_dir or ""
        output_dir = st.text_input("Output Directory", value=default_dir)

        if not output_dir:
            st.info("Enter the path to an output directory containing visualization results.")
            st.stop()

        output_path = Path(output_dir)
        if not output_path.exists():
            st.error(f"Directory not found: {output_path}")
            st.stop()

    # Load results
    results = load_results_directory(output_path)

    # Session state for navigation
    if "selected_direction" not in st.session_state:
        st.session_state.selected_direction = None

    # Render appropriate view
    if st.session_state.selected_direction:
        if render_direction_detail(st.session_state.selected_direction, results):
            st.session_state.selected_direction = None
            st.rerun()
    else:
        selected = render_dashboard(results, output_path)
        if selected:
            st.session_state.selected_direction = selected
            st.rerun()


if __name__ == "__main__":
    main()
