import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from interventionfeatures.core.search import ActivationSimSearcher

# Optional scipy import for confidence intervals
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

# Local imports - using your existing modules
from ..core import data_handler, model
from ..core.data_handler import TransformerDataHandler
from ..core.model import IntervenableTransformerSegment
from ..config import Config
from .explainer import FeatureExplainer

# Optional LangChain imports for explanation generation
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatAnthropic = None
    ChatOpenAI = None
    ChatPromptTemplate = None
    JsonOutputParser = None

# Optional SAE imports for SAE-based faithfulness testing
try:
    from sae_lens import SAE
    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False
    SAE = None


def _find_optimal_threshold(matching_values, non_matching_values) -> float:
    """
    Finds the optimal threshold to separate matching and non-matching example activations.

    The optimal threshold maximizes classification accuracy for determining whether an example
    should activate this feature based on whether it matches the feature's explanation.

    Args:
        matching_values (list): Activation values for examples that should match this feature's explanation.
        non_matching_values (list): Activation values for examples that should NOT match this feature's explanation.

    Returns:
        float: The optimal threshold value.
F    """
    # Convert to lists if tensors are provided
    if hasattr(matching_values, 'tolist'):
        matching_values = matching_values.tolist()
    if hasattr(non_matching_values, 'tolist'):
        non_matching_values = non_matching_values.tolist()
    
    all_activations = matching_values + non_matching_values
    if not all_activations:
        raise ValueError(f"No activations provided: {all_activations}")

    # Get unique, sorted activation values
    unique_activations = sorted(list(set(all_activations)))
    
    best_threshold = unique_activations[0]

    # Create a list of candidate thresholds to test.
    # We test points lower than the min, higher than the max, and all midpoints.
    candidate_thresholds = [unique_activations[0] - 1] 
    candidate_thresholds.extend((unique_activations[i] + unique_activations[i+1]) / 2.0 
                                for i in range(len(unique_activations) - 1))
    candidate_thresholds.append(unique_activations[-1] + 1)
    max_accuracy = 0.0

    # Test each candidate threshold
    for t in candidate_thresholds:
        # True Positives: Matching examples correctly classified (activation > t)
        tp = sum(1 for act in matching_values if act > t)
        # True Negatives: Non-matching examples correctly classified (activation <= t)
        tn = sum(1 for act in non_matching_values if act <= t)
        
        current_accuracy = (tp + tn) / len(all_activations)
        
        # If this threshold gives higher accuracy, update our best
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_threshold = t
            
    return best_threshold

class FaithfulnessConfig:
    """Configuration class for faithfulness testing."""

    def __init__(self, parent_config: Config):
        self.parent_config = parent_config
        self.llm_provider = parent_config.llm.faithfulness.provider
        self.llm_model_name = parent_config.llm.faithfulness.model_name
        self.batch_size = 32
        self.cosine_acts = True
        self.device = self._setup_device()

    def _setup_device(self):
        """Determine and set up the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# --- Part 2: LangChain Setup for Test Data Generation ---

def _get_chat_model(provider: str, model_name: str, temperature: float = 0.0):
    """Returns the chat model instance based on the provider."""
    if provider == "anthropic":
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic is not installed.")
        return ChatAnthropic(model=model_name, temperature=temperature)
    elif provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed.")
        return ChatOpenAI(model=model_name)#, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_test_data_chain(provider: str, model_name: str):
    """Creates a LangChain chain to generate a synthetic test dataset from an explanation."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for faithfulness testing. Install with: pip install langchain langchain-anthropic langchain-openai"
        )

    model = _get_chat_model(provider, model_name)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        You are a research assistant specializing in language model evaluation. I will provide an
        explanation of a feature's behavior. Your task is to generate a JSON object with two lists
        of short, diverse text examples for a faithfulness test:
        1. `matching_examples`: 10 examples that clearly match the explanation.
        2. `non_matching_examples`: 10 examples that do NOT match.

        Return in JSON with keys "matching_examples" and "non_matching_examples". Use basic JSON, no extra comments
        """,
            ),
            ("human", "The explanation for the feature is: {explanation}"),
        ]
    )
    parser = JsonOutputParser()
    return prompt_template | model | parser


def _invoke_with_retry(chain, input_data: Dict[str, Any], max_retries: int = 10, delay: float = 1.0) -> Dict[str, Any]:
    """
    Invoke a LangChain chain with retry logic for parsing failures.
    
    Args:
        chain: The LangChain chain to invoke
        input_data: Input data for the chain
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
    
    Returns:
        Parsed result from the chain
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            result = chain.invoke(input_data)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            # These are likely parsing errors that might be resolved by retrying
            last_exception = e
            if attempt < max_retries:
                retry_delay = delay * (2 ** attempt)  # Exponential backoff
                print(f"LangChain parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries + 1} attempts failed. Last error: {e}")
        except Exception as e:
            # For non-parsing errors, don't retry and re-raise immediately
            print(f"Non-parsing error occurred, not retrying: {e}")
            raise e
    
    # If we get here, all retries failed
    raise last_exception


# --- Part 3: Faithfulness Test Implementation using your codebase ---


def _run_simulation_test(
    css_direction,
    test_data,
    data_handler,
    config: FaithfulnessConfig,
):
    """
    Runs a simulation test to measure the faithfulness of an explanation using your CSS directions.

    Args:
        css_direction (torch.Tensor): The CSS direction vector to test.
        test_data (dict): A dictionary with 'matching_examples' and 'non_matching_examples'.
        data_handler (TransformerDataHandler): Your data handler instance.
        model_segment (IntervenableTransformerSegment): Your model segment for intervention.
        config (FaithfulnessConfig): Configuration object.

    Returns:
        dict: A dictionary containing the accuracy score and other details.
    """
    matching_examples = test_data["matching_examples"]
    non_matching_examples = test_data["non_matching_examples"]

    # Prepare batches and labels
    all_examples = matching_examples + non_matching_examples
    labels = [1] * len(matching_examples) + [0] * len(non_matching_examples)

    print(f"\nRunning faithfulness test on {len(all_examples)} examples...")

    # Tokenize examples using your data handler
    activations = []

    for i, text in enumerate(all_examples):
        # Tokenize and get activations using your existing pipeline
        tokens = data_handler.tokenizer(
            text,
            return_tensors="pt",
            max_length=config.parent_config.model.max_seq_len,
            truncation=True,
            padding=True,
        )["input_ids"].to(config.device)

        # Get activations from your model
        with torch.no_grad():
            _, cache = data_handler.model.run_with_cache(
                tokens,
                names_filter=[
                    data_handler.hook_point_name_for_x,
                ],
            )
            x_batch = cache[data_handler.hook_point_name_for_x]

            # outputs = data_handler.model(**tokens, output_hidden_states=True)
            ## Extract the relevant layer's activations
            # hidden_states = outputs.hidden_states[config.parent_config.layer_cutoff]  # Shape: [Batch size, seq_len, d_model]
            hidden_states = x_batch  
            c = css_direction.unsqueeze(1).unsqueeze(2).to(config.device)
            h = hidden_states.unsqueeze(0)
            # css_direction: [K, d_model]
            css_activations = (
                torch.sum(h * c, dim=-1)
            ) if not config.cosine_acts else F.cosine_similarity(h, c, dim=-1)

            # Take max activation across sequence positions
            max_activation_per_pos = torch.max(css_activations, dim=-1)[
                0
            ]  # Across sequence
            print("QQQQ", max_activation_per_pos.shape, css_activations.shape)
            if max_activation_per_pos.shape[-1] > 1:
                raise NotImplementedError("Scoring for K > 1 not yet implemented")
            assert (
                max_activation_per_pos.shape[-1] == 1
            ), "Batch size only allowed to be 1"

            # Compute dot product with CSS direction to get activation strength
            css_activation = torch.max(max_activation_per_pos).squeeze().item()

        activations.append(css_activation)

    # Calculate predictions and accuracy
    results = []

    for i, activation_val in enumerate(activations):
        # Prediction based on threshold
        results.append(
            {
                "text": all_examples[i],
                "label": labels[i],
                #"prediction": prediction, # These will get filled in later
                "activation": activation_val,
                #"correct": is_correct, # These will get filled in later
            }
        )

    return {
        #"accuracy": accuracy, # Will get filled in later
        "results": results
        }

def get_sae_directions_and_explanations(
    config: MainConfig,
    model: IntervenableTransformerSegment,
    data_handler: TransformerDataHandler,
    searcher: ActivationSimSearcher,
    feature_indices: torch.tensor,
    top_k_activations: int = 10,
) -> Tuple[List[Dict], List[Dict], List[float]]:
    """
    Extract SAE feature directions and generate explanations for them.
    
    Args:
        config: Configuration containing SAE model name and layer info
        model: IntervenableTransformerSegment instance
        data_handler: Data handler for tokenization and activation extraction
        searcher: ActivationSimSearcher with indexed dataset for finding examples
        feature_indices: Features of the SAE to use
        top_k_activations: Number of top activating examples to use for explanation
        
    Returns:
        Tuple of (sae_results, explanations, max_activations)
    """
    if not SAE_AVAILABLE:
        raise ImportError(
            "sae_lens is required for SAE faithfulness testing. Install with: pip install sae_lens"
        )
    
    if not hasattr(config, 'model_name_sae') or not config.model_name_sae:
        raise ValueError("config.model_name_sae must be specified for SAE faithfulness testing")
    
    print(f"Loading SAE '{config.model_name_sae}' for layer {config.layer_cutoff}...")
    
    # Load SAE
    sae, _, _ = SAE.from_pretrained(
        release=config.model_name_sae,
        sae_id=f"blocks.{config.layer_cutoff}.{config.hook_type}",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Get SAE encoder directions (transposed to get [n_features, d_model])
    sae_directions = sae.W_enc.T
    
    selected_directions = sae_directions[feature_indices]
    
    print(f"Selected {feature_indices.shape[0]} SAE feature directions from {sae_directions.shape[0]} available")
    
    # Create SAE results format compatible with existing faithfulness testing
    sae_results = []
    explanations = []
    max_activations = []
    
    # Import explainer for generating explanations
    from .explainer import FeatureExplainer
    
    # Create FeatureExplainer instance to use the proper explanation generation
    explainer = FeatureExplainer(
        model=model,
        explainer_llm_provider=config.faithfulness_llm_provider,
        explainer_llm_model_name=config.faithfulness_llm_model_name,
    )
    
    # Note: We now use the ActivationSimSearcher's indexed dataset instead of sample texts
    # This leverages the same large dataset that CSS directions use for finding examples
    
    for i, (feature_idx, direction) in enumerate(zip(feature_indices, selected_directions)):
        print(f"Processing SAE feature {i+1}/{feature_indices.shape[0]} (feature index {feature_idx})")
        
        # Create a CSS-like result structure
        sae_result = {
            "s": direction.unsqueeze(0),  # The direction vector
            "score_J_s": 0.0,  # SAE features don't have CSS scores, so set to 0
            "feature_idx": feature_idx.item(),
            "feature_type": "sae",
            "presence_threshold": sae.b_enc[feature_idx].item()  - torch.inner(sae.b_dec, direction.squeeze()).item() # The cutoff to grt passed the ReLU
        }
        
        # Generate explanation using the proper FeatureExplainer with SAE activations
        try:
            # Use the explainer to generate high-quality explanations like CSS directions
            explanation_text, top_examples, max_activation = explainer.explain_sae_feature(
                sae=sae,
                feature_idx=feature_idx.item(),
                searcher=searcher,
                data_handler=data_handler,
                num_examples=top_k_activations,
            )
            
            print(f"Generated explanation for SAE feature {feature_idx}: {explanation_text[:100]}...")
            
        except Exception as e:
            raise e
            print(f"Warning: Could not generate explanation using explainer for SAE feature {feature_idx}: {e}")
            # Fallback to simple explanation
            max_activation = torch.linalg.norm(direction).item()
            explanation_text = f"SAE Feature {feature_idx.item()}: Feature extracted from layer {config.layer_cutoff} {config.hook_type} activations from SAE '{config.model_name_sae}'. Error during explanation generation: {str(e)}"
        
        explanation = {
            "explanation": explanation_text,
            "feature_idx": feature_idx.item(),
            "feature_type": "sae",
        }
        
        sae_results.append(sae_result)
        explanations.append(explanation)
        max_activations.append(max_activation)
    
    return sae_results, explanations, max_activations


def run_multi_trial_faithfulness_testing(
    css_results: List[Dict],
    explanations: List[Dict],
    data_handler: TransformerDataHandler,
    config,
    total_trials: int = 5,
    output_dir: Optional[Path] = None,
    progress=None,
    thresholds: Optional[List[float]] = None,
    use_cosine_sim = True
) -> Dict[str, Any]:
    """
    Run multiple trials of faithfulness testing on CSS directions and explanations.

    Args:
        css_results: List of CSS direction results
        explanations: List of explanation dictionaries
        data_handler: TransformerDataHandler instance
        config: Main configuration object
        total_trials: Number of trials to run
        output_dir: Directory to save HTML visualizations
        progress: Optional progress tracker
        combined_html: If True, generate combined HTML file; if False, generate individual trial files

    Returns:
        Dictionary containing aggregated results and statistics
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for faithfulness testing. Install with: pip install langchain langchain-anthropic langchain-openai"
        )

    faithfulness_config = FaithfulnessConfig(config
                                             )
    faithfulness_config.cosine_acts = use_cosine_sim

    # Ensure API key is set
    if faithfulness_config.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    elif faithfulness_config.llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")


    # Setup LangChain test data generation
    test_data_chain = create_test_data_chain(faithfulness_config.llm_provider, faithfulness_config.llm_model_name)

    # Initialize results storage
    all_results = []
    direction_summaries = []

    # Extract explanations text from the explanations data
    explanation_texts = []
    for exp_item in explanations:
        if isinstance(exp_item, dict) and "explanation" in exp_item:
            explanation_texts.append(exp_item["explanation"])
        else:
            explanation_texts.append(str(exp_item))

    # Limit to the minimum of CSS results and explanations
    num_directions = min(len(css_results), len(explanation_texts))

    if progress:
        task = progress.add_task("[cyan]Doing Faithfulness...", total=num_directions)
    for direction_idx in range(num_directions):
        css_result = css_results[direction_idx]
        explanation_text = explanation_texts[direction_idx]

        if progress:
            progress.update(
                task,
                description=f"Testing direction {direction_idx + 1}/{num_directions}...",
            )

        # Storage for this direction across all trials
        direction_trials = []
        css_direction = (
            css_result["s"] if isinstance(css_result, dict) else css_result.s
        )
        css_score = (
            css_result.get("score_J_s", 0.0)
            if isinstance(css_result, dict)
            else getattr(css_result, "score_J_s", 0.0)
        )

        for trial_idx in range(total_trials):
            if progress:
                progress.update(
                    progress.tasks[0].id,
                    description=f"Direction {direction_idx + 1}/{num_directions}, Trial {trial_idx + 1}/{total_trials}...",
                )

            try:
                # Generate test data for this trial
                test_data = _invoke_with_retry(test_data_chain, {"explanation": explanation_text})

                # Run faithfulness test
                trial_result = _run_simulation_test(
                    css_direction=css_direction,
                    test_data=test_data,
                    data_handler=data_handler,
                    config=faithfulness_config,
                )

                # Add trial metadata
                trial_result.update(
                    {
                        "trial_idx": trial_idx,
                        "direction_idx": direction_idx,
                        "css_score": css_score,
                        "explanation": explanation_text,
                        "test_data": test_data,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                direction_trials.append(trial_result)

            except Exception as e:
                print(f"Trial {trial_idx} failed for direction {direction_idx}: {e}")
                # Add failed trial placeholder
                failed_trial = {
                    "trial_idx": trial_idx,
                    "direction_idx": direction_idx,
                    "css_score": css_score,
                    "explanation": explanation_text,
                    "accuracy": 0.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                direction_trials.append(failed_trial)

        # First, determine the optimal threshold and compute final predictions
        matching_values = []
        non_matching_values = []
        print(direction_trials)
        for d in direction_trials:
            if 'results' in d:  # Only process successful trials
                matching_values += [x['activation'] for x in d['results'] if x['label'] == 1]
                non_matching_values += [x['activation'] for x in d['results'] if x['label'] == 0]
            
        if thresholds:
            threshold = thresholds[direction_idx]
        else:
            threshold = _find_optimal_threshold(matching_values, non_matching_values)

        # Apply threshold and compute final accuracy for each trial
        n_corr = 0
        n_tot = 0
        for i in range(len(direction_trials)):
            if 'results' in direction_trials[i]:  # Only process successful trials
                n_corr_i = 0
                n_tot_i = len(direction_trials[i]['results'])
                for j in range(len(direction_trials[i]['results'])):
                    act = direction_trials[i]['results'][j]['activation']
                    direction_trials[i]['results'][j]['prediction'] = 1 if act > threshold else 0
                    is_correct = direction_trials[i]['results'][j]['prediction']  == direction_trials[i]['results'][j]['label'] 
                    direction_trials[i]['results'][j]['correct'] = is_correct
                    n_corr_i += is_correct
                direction_trials[i]['accuracy'] = n_corr_i / n_tot_i
                
                n_corr += n_corr_i
                n_tot += n_tot_i

        # Now compute statistics for this direction across trials with correct accuracies
        accuracies = [
            trial["accuracy"] for trial in direction_trials if "accuracy" in trial
        ]

        if accuracies:
            direction_summary = {
                "direction_idx": direction_idx,
                "css_score": css_score,
                "explanation": explanation_text,
                "num_trials": len(direction_trials),
                "num_successful_trials": len(accuracies),
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "median_accuracy": np.median(accuracies),
                "accuracy_95_ci": _compute_confidence_interval(
                    accuracies, confidence=0.95
                ),
                "optimal_threshold": threshold,
                "trials": direction_trials,
            }
        else:
            direction_summary = {
                "direction_idx": direction_idx,
                "css_score": css_score,
                "explanation": explanation_text,
                "num_trials": len(direction_trials),
                "num_successful_trials": 0,
                "mean_accuracy": 0.0,
                "optimal_threshold": threshold,
                "error": "All trials failed",
                "trials": direction_trials,
            }

        direction_summaries.append(direction_summary)
        all_results.extend(direction_trials)

    # Compute overall statistics
    all_accuracies = [trial["accuracy"] for trial in all_results if "accuracy" in trial]

    overall_summary = {
        "num_directions": num_directions,
        "total_trials": len(all_results),
        "total_successful_trials": len(all_accuracies),
        "overall_mean_accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
        "overall_std_accuracy": np.std(all_accuracies) if all_accuracies else 0.0,
        "overall_accuracy_95_ci": _compute_confidence_interval(
            all_accuracies, confidence=0.95
        )
        if all_accuracies
        else None,
    }
    
    # Compute feature group statistics (positive vs negative polarity groups)
    feature_group_stats = _compute_feature_group_statistics(direction_summaries, confidence=0.95)
    overall_summary["feature_group_statistics"] = feature_group_stats

    # Generate HTML visualizations if output_dir is provided
    if output_dir:
        # Create html subdirectory for consolidated visualization files
        html_dir = output_dir / "html"
        html_dir.mkdir(exist_ok=True)
        
        # Always generate summary HTML
        summary_html_file = html_dir / "faithfulness_summary.html"
        _generate_summary_html(
            direction_summaries, overall_summary, str(summary_html_file),
            sort_by="mean_accuracy"
        )
        
        # Generate combined HTML if requested
        combined_html_file = html_dir / "faithfulness_combined.html"
        _generate_combined_faithfulness_html(
            all_results, direction_summaries, str(combined_html_file)
        )

    return {
        "overall_summary": overall_summary,
        "direction_summaries": direction_summaries,
        "all_trials": all_results,
        "parameters": {
            "total_trials": total_trials,
            "num_directions_tested": num_directions,
        },
    }


def run_sae_faithfulness_testing(
    config: MainConfig,
    data_handler: TransformerDataHandler,
    model: IntervenableTransformerSegment,
    num_features: int = 20,
    total_trials: int = 5,
    output_dir: Optional[Path] = None,
    progress=None,
    searcher: ActivationSimSearcher = None,
) -> Dict[str, Any]:
    """
    Run faithfulness testing specifically for SAE features.
    
    This function extracts SAE feature directions, generates explanations for them,
    and runs the standard faithfulness testing pipeline.
    
    Args:
        config: Configuration containing SAE model name and other parameters
        data_handler: TransformerDataHandler instance
        model: IntervenableTransformerSegment instance
        num_features: Number of SAE features to test
        total_trials: Number of trials to run for each feature
        output_dir: Directory to save HTML visualizations
        progress: Optional progress tracker
        searcher: Optional ActivationSimSearcher with indexed data for finding examples.
                 If None, will create a basic searcher or fall back to sample texts.
        
    Returns:
        Dictionary containing faithfulness testing results
    """
    if not SAE_AVAILABLE:
        raise ImportError(
            "sae_lens is required for SAE faithfulness testing. Install with: pip install sae_lens"
        )
    
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for faithfulness testing. Install with: pip install langchain langchain-anthropic langchain-openai"
        )
    
    print(f"🧠 Starting SAE faithfulness testing with {num_features} features...")
    
    # Extract SAE directions and generate explanations
    if searcher is None:
        # If no searcher provided, we need to create one or use a fallback approach
        # For now, raise an informative error suggesting the user provide a searcher
        raise ValueError(
            "A searcher with indexed activations is required for SAE faithfulness testing. "
            "Please provide a searcher that has been built with a dataset using the same "
            "hook point as the SAE analysis. You can create one using ActivationSimSearcher "
            "and index it with your dataset, or use the standard CSS pipeline which creates "
            "a searcher automatically."
        )

    # Load SAE
    sae, _, _ = SAE.from_pretrained(
        release=config.model_name_sae,
        sae_id=f"blocks.{config.layer_cutoff}.{config.hook_type}",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    sae_directions = sae.W_enc.T
    # Randomly sample features to evaluate
    if num_features > sae_directions.shape[0]:
        num_features = sae_directions.shape[0]
        print(f"Warning: Requested {num_features} features but SAE only has {sae_directions.shape[0]}. Using all available.")
    #feature_indices = torch.tensor([2])
    feature_indices = torch.randperm(sae_directions.shape[0])[:num_features]
    
    sae_results, explanations, max_activations = get_sae_directions_and_explanations(
        config=config,
        model=model,
        data_handler=data_handler,
        searcher=searcher,
        feature_indices=feature_indices
    )
    thresholds = [
        s['presence_threshold'] for s in sae_results
    ]
    
    print(f"✅ Extracted {len(sae_results)} SAE features and generated explanations")
    
    # Run the standard faithfulness testing pipeline
    results = run_multi_trial_faithfulness_testing(
        css_results=sae_results,
        explanations=explanations,
        data_handler=data_handler,
        config=config,
        total_trials=total_trials,
        output_dir=output_dir,
        progress=progress,
        thresholds=thresholds,
        use_cosine_sim=False
    )
    
    # Add SAE-specific metadata
    results["sae_metadata"] = {
        "model_name_sae": config.model_name_sae,
        "layer_cutoff": config.layer_cutoff,
        "hook_type": config.hook_type,
        "num_features_tested": num_features,
        "feature_type": "sae",
    }
    
    return results


def _compute_confidence_interval(
    data: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for a list of values."""
    if not data:
        return (0.0, 0.0)

    data_array = np.array(data)
    n = len(data_array)
    mean = np.mean(data_array)
    sem = np.std(data_array) / np.sqrt(n)  # Standard error of the mean

    if SCIPY_AVAILABLE and n > 1:
        # Use t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * sem
    else:
        # Fallback to normal approximation if scipy not available
        # Using 1.96 for 95% confidence interval
        z_value = 1.96 if confidence == 0.95 else 2.58  # rough approximation for 99%
        margin_error = z_value * sem

    return (mean - margin_error, mean + margin_error)


def _compute_feature_group_statistics(
    direction_summaries: List[Dict], confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Compute aggregated statistics grouped by feature polarity.
    
    Args:
        direction_summaries: List of direction summary dictionaries with polarity info
        confidence: Confidence level for intervals (default 0.95)
        
    Returns:
        Dictionary containing positive, negative, and combined group statistics
    """
    positive_features = [d for d in direction_summaries if d.get('polarity') == 'positive']
    negative_features = [d for d in direction_summaries if d.get('polarity') == 'negative']
    
    def _compute_group_stats(features, group_name):
        if not features:
            return {
                'count': 0,
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'median_accuracy': 0.0,
                'min_accuracy': 0.0,
                'max_accuracy': 0.0,
                'accuracy_95_ci': (0.0, 0.0),
                'features': []
            }
        
        accuracies = [f.get('mean_accuracy', 0.0) for f in features]
        
        return {
            'count': len(features),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'median_accuracy': np.median(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_95_ci': _compute_confidence_interval(accuracies, confidence),
            'features': features
        }
    
    positive_stats = _compute_group_stats(positive_features, 'positive')
    negative_stats = _compute_group_stats(negative_features, 'negative')
    
    # Combined statistics across all features
    all_features = direction_summaries
    all_accuracies = [f.get('mean_accuracy', 0.0) for f in all_features]
    combined_stats = {
        'count': len(all_features),
        'mean_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
        'std_accuracy': np.std(all_accuracies) if all_accuracies else 0.0,
        'median_accuracy': np.median(all_accuracies) if all_accuracies else 0.0,
        'min_accuracy': np.min(all_accuracies) if all_accuracies else 0.0,
        'max_accuracy': np.max(all_accuracies) if all_accuracies else 0.0,
        'accuracy_95_ci': _compute_confidence_interval(all_accuracies, confidence) if all_accuracies else (0.0, 0.0),
    }
    
    # Comparison statistics between positive and negative groups
    comparison_stats = {}
    if positive_features and negative_features:
        pos_accuracies = [f.get('mean_accuracy', 0.0) for f in positive_features]
        neg_accuracies = [f.get('mean_accuracy', 0.0) for f in negative_features]
        
        comparison_stats = {
            'positive_vs_negative_diff': positive_stats['mean_accuracy'] - negative_stats['mean_accuracy'],
            'positive_vs_negative_diff_abs': abs(positive_stats['mean_accuracy'] - negative_stats['mean_accuracy']),
        }
        
        # Statistical significance test if scipy available and sufficient data
        if SCIPY_AVAILABLE and len(pos_accuracies) > 1 and len(neg_accuracies) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(pos_accuracies, neg_accuracies)
                comparison_stats.update({
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant_at_05': p_value < 0.05,
                    'significant_at_01': p_value < 0.01
                })
            except Exception:
                # In case of any statistical test errors
                pass
    
    return {
        'positive_group': positive_stats,
        'negative_group': negative_stats,
        'combined': combined_stats,
        'group_comparison': comparison_stats
    }




def _generate_faithfulness_html(trial_result: Dict[str, Any], output_file: str) -> None:
    """Generate HTML visualization for a single faithfulness trial."""
    direction_idx = trial_result["direction_idx"]
    trial_idx = trial_result["trial_idx"]
    accuracy = trial_result["accuracy"]
    explanation = trial_result["explanation"]
    css_score = trial_result["css_score"]

    # Extract detailed results
    detailed_results = trial_result.get("results", [])
    test_data = trial_result.get("test_data", {})

    # Calculate statistics
    total_examples = len(detailed_results)
    correct_predictions = sum(
        1 for res in detailed_results if res.get("correct", False)
    )
    matching_examples = [res for res in detailed_results if res.get("label") == 1]
    non_matching_examples = [res for res in detailed_results if res.get("label") == 0]

    matching_correct = sum(1 for res in matching_examples if res.get("correct", False))
    non_matching_correct = sum(1 for res in non_matching_examples if res.get("correct", False))

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faithfulness Test - Direction {direction_idx} Trial {trial_idx}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #212529;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        }}
        h1 {{
            color: #0056b3;
            text-align: center;
            margin-bottom: 30px;
            font-weight: 300;
        }}
        h2 {{
            color: #0056b3;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border-left: 4px solid #0056b3;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #0056b3;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }}
        .explanation-box {{
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }}
        .example-section {{
            margin: 20px 0;
        }}
        .example {{
            margin: 10px 0;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #28a745;
        }}
        .example.incorrect {{
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }}
        .example.correct {{
            background-color: #d4edda;
        }}
        .example-text {{
            font-weight: 500;
            margin-bottom: 8px;
        }}
        .example-details {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        .activation-viz {{
            display: inline-block;
            width: 100px;
            height: 20px;
            background: linear-gradient(90deg, #fff 0%, #007bff 100%);
            border: 1px solid #dee2e6;
            border-radius: 3px;
            position: relative;
        }}
        .activation-marker {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background-color: #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Faithfulness Test Results</h1>

        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{direction_idx}</div>
                <div class="stat-label">Direction Index</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{trial_idx}</div>
                <div class="stat-label">Trial Number</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{accuracy:.1%}</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{css_score:.4f}</div>
                <div class="stat-label">CSS Score</div>
            </div>
        </div>

        <div class="explanation-box">
            <h3 style="margin-top: 0; color: #1565c0;">Feature Explanation</h3>
            <p>{explanation}</p>
        </div>

        <h2>Performance Summary</h2>
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{correct_predictions}/{total_examples}</div>
                <div class="stat-label">Correct Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{matching_correct}/{len(matching_examples)}</div>
                <div class="stat-label">Matching Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{non_matching_correct}/{len(non_matching_examples)}</div>
                <div class="stat-label">Non-Matching Accuracy</div>
            </div>
        </div>

        <h2>Matching Examples</h2>
        <div class="example-section">
"""

    # Add matching examples
    for example in matching_examples:
        correct_class = "correct" if example.get("correct", False) else "incorrect"
        activation = example.get("activation", 0.0)
        prediction = example.get("prediction", 0)

        html_content += f"""
            <div class="example {correct_class}">
                <div class="example-text">"{example.get('text', '')[:200]}..."</div>
                <div class="example-details">
                    Prediction: {"Positive" if prediction else "Negative"} |
                    Activation: {activation:.3f} |
                    Result: {"✓ Correct" if example.get("correct", False) else "✗ Incorrect"}
                </div>
            </div>
"""

    html_content += """
        </div>

        <h2>Non-Matching Examples</h2>
        <div class="example-section">
"""

    # Add non-matching examples
    for example in non_matching_examples:
        correct_class = "correct" if example.get("correct", False) else "incorrect"
        activation = example.get("activation", 0.0)
        prediction = example.get("prediction", 0)

        html_content += f"""
            <div class="example {correct_class}">
                <div class="example-text">"{example.get('text', '')[:200]}..."</div>
                <div class="example-details">
                    Prediction: {"Positive" if prediction else "Negative"} |
                    Activation: {activation:.3f} |
                    Result: {"✓ Correct" if example.get("correct", False) else "✗ Incorrect"}
                </div>
            </div>
"""

    html_content += """
        </div>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 0.9em;">
            Generated by FourierFeatures Faithfulness Testing
        </div>
    </div>
</body>
</html>
"""

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Warning: Failed to write HTML file {output_file}: {e}")


def _generate_combined_faithfulness_html(
    all_trial_results: List[Dict], direction_summaries: List[Dict], output_file: str
) -> None:
    """Generate combined HTML visualization for all faithfulness trials."""
    # Group trials by direction
    direction_trials = {}
    for trial in all_trial_results:
        direction_idx = trial["direction_idx"]
        if direction_idx not in direction_trials:
            direction_trials[direction_idx] = []
        direction_trials[direction_idx].append(trial)
    
    # Sort directions by index
    sorted_directions = sorted(direction_trials.keys())
    
    # Generate trial cards HTML
    trial_cards_html = ""
    for direction_idx in sorted_directions:
        trials = direction_trials[direction_idx]
        direction_summary = next((s for s in direction_summaries if s["direction_idx"] == direction_idx), {})
        
        # Direction header
        mean_accuracy = direction_summary.get("mean_accuracy", 0.0)
        optimal_threshold = direction_summary.get("optimal_threshold", "N/A")
        threshold_display = f"{optimal_threshold:.3f}" if isinstance(optimal_threshold, (int, float)) else str(optimal_threshold)
        num_trials = len(trials)
        
        trial_cards_html += f"""
        <div class="direction-section">
            <div class="direction-header" onclick="toggleDirection({direction_idx})">
                <h2>Direction {direction_idx}</h2>
                <div class="direction-stats">
                    <span class="stat">Trials: {num_trials}</span>
                    <span class="stat">Avg Accuracy: {mean_accuracy:.1%}</span>
                    <span class="stat">Threshold: {threshold_display}</span>
                </div>
                <span class="toggle-icon rotated" id="toggle-{direction_idx}">▼</span>
            </div>
            <div class="direction-content hidden" id="direction-{direction_idx}">
        """
        
        # Trial cards for this direction
        for trial in trials:
            trial_idx = trial["trial_idx"]
            accuracy = trial.get("accuracy", 0.0)
            explanation = trial.get("explanation", "No explanation available")
            css_score = trial.get("css_score", 0.0)
            
            # Get detailed results for examples
            detailed_results = trial.get("results", [])
            total_examples = len(detailed_results)
            correct_predictions = sum(1 for res in detailed_results if res.get("correct", False))
            
            # Generate examples HTML with clearer display
            examples_html = ""
            threshold_val = optimal_threshold if isinstance(optimal_threshold, (int, float)) else 0.0
            
            for i, result in enumerate(detailed_results):  # Show all examples
                correct = result.get("correct", False)
                label = result.get("label", "Unknown")
                prediction = result.get("prediction", "Unknown")
                activation = result.get("activation", 0.0)
                text = result.get("text", "")
                
                status_class = "correct" if correct else "incorrect"
                status_icon = "✓" if correct else "✗"
                label_text = "Positive" if label == 1 else "Negative"
                pred_text = "Positive" if prediction == 1 else "Negative"
                
                # Show relationship to threshold
                threshold_relation = ""
                if isinstance(threshold_val, (int, float)):
                    if activation > threshold_val:
                        threshold_relation = f"(>{threshold_val:.3f})"
                    else:
                        threshold_relation = f"(<={threshold_val:.3f})"
                
                examples_html += f"""
                <div class="example-detailed {status_class}">
                    <div class="example-header">
                        <span class="example-num">#{i+1}</span>
                        <span class="example-status">{status_icon} {status_class.title()}</span>
                        <span class="example-activation">Activation: {activation:.3f} {threshold_relation}</span>
                    </div>
                    <div class="example-text">"{text}"</div>
                    <div class="example-labels">
                        <span class="label-actual">True Label: {label_text}</span>
                        <span class="label-predicted">Predicted: {pred_text}</span>
                    </div>
                </div>
                """
            
            trial_cards_html += f"""
            <div class="trial-card">
                <div class="trial-header">
                    <h3>Trial {trial_idx}</h3>
                    <div class="trial-stats">
                        <span class="stat accuracy">Accuracy: {accuracy:.1%}</span>
                        <span class="stat">CSS Score: {css_score:.3f}</span>
                        <span class="stat">Examples: {correct_predictions}/{total_examples}</span>
                        <span class="stat">Threshold: {threshold_display}</span>
                    </div>
                </div>
                <div class="trial-explanation">
                    <h4>Explanation:</h4>
                    <p>{explanation}</p>
                </div>
                <div class="trial-examples">
                    <h4>Test Results (Activation > {threshold_display} = Positive):</h4>
                    <div class="examples-detailed-grid">
                        {examples_html}
                    </div>
                </div>
            </div>
            """
        
        trial_cards_html += """
            </div>
        </div>
        """
    
    # Navigation menu
    nav_html = ""
    for direction_idx in sorted_directions:
        nav_html += f'<a href="#direction-{direction_idx}" class="nav-link">Direction {direction_idx}</a>'
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Faithfulness Testing Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #212529;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        }}
        h1 {{
            color: #0056b3;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }}
        .navigation {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .nav-link {{
            display: inline-block;
            padding: 8px 16px;
            margin: 0 5px;
            background-color: #0056b3;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }}
        .nav-link:hover {{
            background-color: #004494;
        }}
        .direction-section {{
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            overflow: hidden;
        }}
        .direction-header {{
            background-color: #0056b3;
            color: white;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.3s;
        }}
        .direction-header:hover {{
            background-color: #004494;
        }}
        .direction-header h2 {{
            margin: 0;
            font-size: 1.5rem;
        }}
        .direction-stats {{
            display: flex;
            gap: 20px;
        }}
        .stat {{
            padding: 5px 10px;
            background-color: rgba(255,255,255,0.2);
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        .stat.accuracy {{
            background-color: #28a745;
        }}
        .toggle-icon {{
            font-size: 1.2rem;
            transition: transform 0.3s;
        }}
        .toggle-icon.rotated {{
            transform: rotate(180deg);
        }}
        .direction-content {{
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .direction-content.hidden {{
            display: none;
        }}
        .trial-card {{
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .trial-header {{
            background-color: #e9ecef;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .trial-header h3 {{
            margin: 0;
            color: #0056b3;
        }}
        .trial-stats {{
            display: flex;
            gap: 15px;
        }}
        .trial-explanation {{
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
        }}
        .trial-explanation h4 {{
            margin: 0 0 10px 0;
            color: #0056b3;
        }}
        .trial-examples {{
            padding: 15px;
        }}
        .trial-examples h4 {{
            margin: 0 0 10px 0;
            color: #0056b3;
        }}
        .examples-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }}
        .example {{
            padding: 8px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
        }}
        .example.correct {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }}
        .example.incorrect {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
        }}
        .example-num {{
            font-weight: bold;
            color: #495057;
        }}
        .examples-note {{
            margin-top: 10px;
            font-style: italic;
            color: #6c757d;
        }}
        .examples-detailed-grid {{
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-top: 15px;
        }}
        .example-detailed {{
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            background-color: #f8f9fa;
        }}
        .example-detailed.correct {{
            border-left: 4px solid #28a745;
            background-color: #d4edda;
        }}
        .example-detailed.incorrect {{
            border-left: 4px solid #dc3545;
            background-color: #f8d7da;
        }}
        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        .example-status {{
            color: #495057;
        }}
        .example-activation {{
            color: #0056b3;
            font-family: monospace;
        }}
        .example-text {{
            background-color: #ffffff;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-style: italic;
            border-left: 3px solid #dee2e6;
        }}
        .example-labels {{
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
        }}
        .label-actual {{
            color: #495057;
        }}
        .label-predicted {{
            color: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Combined Faithfulness Testing Results</h1>
        
        <div class="navigation">
            {nav_html}
        </div>
        
        {trial_cards_html}
    </div>
    
    <script>
        function toggleDirection(directionIdx) {{
            const content = document.getElementById('direction-' + directionIdx);
            const toggle = document.getElementById('toggle-' + directionIdx);
            
            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                toggle.classList.remove('rotated');
                toggle.textContent = '▼';
            }} else {{
                content.classList.add('hidden');
                toggle.classList.add('rotated');
                toggle.textContent = '▲';
            }}
        }}
        
        // Initialize all sections as expanded
        document.addEventListener('DOMContentLoaded', function() {{
            // All sections start expanded by default
        }});
    </script>
</body>
</html>
"""

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Warning: Failed to write combined HTML file {output_file}: {e}")


def _generate_summary_html(
    direction_summaries: List[Dict], overall_summary: Dict, output_file: str,
    sort_by: str = "mean_accuracy"
) -> None:
    """Generate HTML summary visualization for all faithfulness tests."""
    
    # Sort directions by faithfulness score
    sorted_directions = sort_directions_by_faithfulness(
        direction_summaries, sort_by=sort_by, ascending=False
    )
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faithfulness Testing Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #212529;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        }}
        h1 {{
            color: #0056b3;
            text-align: center;
            margin-bottom: 30px;
            font-weight: 300;
        }}
        h2 {{
            color: #0056b3;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .overall-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border-left: 4px solid #0056b3;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #0056b3;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }}
        .direction-card {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin: 15px 0;
            padding: 20px;
            background-color: #ffffff;
        }}
        .direction-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .direction-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #0056b3;
        }}
        .direction-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .mini-stat {{
            text-align: center;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .mini-stat-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #0056b3;
        }}
        .mini-stat-label {{
            font-size: 0.8em;
            color: #6c757d;
        }}
        .explanation-preview {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Faithfulness Testing Summary</h1>

        <h2>Overall Results</h2>
        <div class="overall-stats">
            <div class="stat-card">
                <div class="stat-value">{overall_summary['num_directions']}</div>
                <div class="stat-label">Directions Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall_summary['total_successful_trials']}/{overall_summary['total_trials']}</div>
                <div class="stat-label">Successful Trials</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall_summary['overall_mean_accuracy']:.1%}</div>
                <div class="stat-label">Mean Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">±{overall_summary['overall_std_accuracy']:.1%}</div>
                <div class="stat-label">Std Deviation</div>
            </div>
        </div>
"""

    if overall_summary.get("overall_accuracy_95_ci"):
        ci_lower, ci_upper = overall_summary["overall_accuracy_95_ci"]
        html_content += f"""
        <div style="text-align: center; margin: 20px 0; color: #6c757d;">
            95% Confidence Interval: [{ci_lower:.1%}, {ci_upper:.1%}]
        </div>
"""

    html_content += f"""
        <h2>Direction-by-Direction Results</h2>
        <div style="text-align: center; margin-bottom: 20px; color: #6c757d; font-style: italic;">
            Directions sorted by {sort_by.replace('_', ' ')} (highest first)
        </div>
"""

    # Add each direction's results (now sorted)
    for rank, direction in enumerate(sorted_directions, 1):
        direction_idx = direction["direction_idx"]
        css_score = direction["css_score"]
        explanation = direction["explanation"]
        mean_accuracy = direction.get("mean_accuracy", 0.0)
        std_accuracy = direction.get("std_accuracy", 0.0)
        num_trials = direction["num_trials"]
        num_successful = direction["num_successful_trials"]

        optimal_threshold = direction.get("optimal_threshold", "N/A")
        threshold_display = f"{optimal_threshold:.3f}" if isinstance(optimal_threshold, (int, float)) else str(optimal_threshold)
        
        html_content += f"""
        <div class="direction-card">
            <div class="direction-header">
                <div class="direction-title">#{rank} - Direction {direction_idx}</div>
                <div style="color: #6c757d;">CSS Score: {css_score:.4f} | Threshold: {threshold_display}</div>
            </div>

            <div class="direction-stats">
                <div class="mini-stat">
                    <div class="mini-stat-value">{mean_accuracy:.1%}</div>
                    <div class="mini-stat-label">Mean Accuracy</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-stat-value">±{std_accuracy:.1%}</div>
                    <div class="mini-stat-label">Std Dev</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-stat-value">{num_successful}/{num_trials}</div>
                    <div class="mini-stat-label">Successful Trials</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-stat-value">{threshold_display}</div>
                    <div class="mini-stat-label">Classification Threshold</div>
                </div>
"""

        if "accuracy_95_ci" in direction:
            ci_lower, ci_upper = direction["accuracy_95_ci"]
            html_content += f"""
                <div class="mini-stat">
                    <div class="mini-stat-value">[{ci_lower:.1%}, {ci_upper:.1%}]</div>
                    <div class="mini-stat-label">95% CI</div>
                </div>
"""

        html_content += f"""
            </div>

            <div class="explanation-preview">
                <strong>Explanation:</strong> {explanation[:200]}{'...' if len(explanation) > 200 else ''}
            </div>
        </div>
"""

    html_content += """
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 0.9em;">
            Generated by FourierFeatures Faithfulness Testing
        </div>
    </div>
</body>
</html>
"""

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Warning: Failed to write summary HTML file {output_file}: {e}")


def run_aggregated_faithfulness_visualization(
    multi_run_results: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    sort_by: str = "aggregated_accuracy",
) -> Dict[str, Any]:
    """
    Aggregate multiple runs of faithfulness testing and visualize as unified dataset.
    
    This function takes multiple results from run_multi_trial_faithfulness_testing
    and combines all individual test results into a single aggregated dataset.
    Instead of separate 10 positive + 10 negative per trial, this creates
    unified pools like 50 positive + 50 negative across 5 runs.
    
    Args:
        multi_run_results: List of results from run_multi_trial_faithfulness_testing
        output_dir: Directory to save aggregated HTML visualization
        
    Returns:
        Dictionary containing aggregated results and statistics
    """
    if not multi_run_results:
        raise ValueError("No multi-run results provided for aggregation")
    
    # Extract all direction indices across all runs
    all_direction_indices = set()
    for run_result in multi_run_results:
        for direction_summary in run_result["direction_summaries"]:
            all_direction_indices.add(direction_summary["direction_idx"])
    
    sorted_direction_indices = sorted(all_direction_indices)
    
    # Aggregate results by direction
    aggregated_directions = []
    
    for direction_idx in sorted_direction_indices:
        # Collect all trials for this direction across all runs
        all_trials_for_direction = []
        direction_metadata = None
        
        for run_idx, run_result in enumerate(multi_run_results):
            # Find trials for this direction in this run
            run_trials = [trial for trial in run_result["all_trials"] 
                         if trial["direction_idx"] == direction_idx]
            
            # Add run metadata to each trial
            for trial in run_trials:
                trial["source_run_idx"] = run_idx
                trial["source_run_timestamp"] = run_result.get("timestamp", "unknown")
            
            all_trials_for_direction.extend(run_trials)
            
            # Extract direction metadata (explanation, CSS score) from first available
            if direction_metadata is None:
                direction_summary = next(
                    (ds for ds in run_result["direction_summaries"] 
                     if ds["direction_idx"] == direction_idx), 
                    None
                )
                if direction_summary:
                    direction_metadata = {
                        "explanation": direction_summary["explanation"],
                        "css_score": direction_summary["css_score"],
                    }
        
        if not all_trials_for_direction:
            continue
            
        # Aggregate all individual test results across all trials
        all_matching_results = []
        all_non_matching_results = []
        
        for trial in all_trials_for_direction:
            if "results" not in trial:
                continue
                
            for result in trial["results"]:
                # Add trial and run metadata to each individual result
                enhanced_result = result.copy()
                enhanced_result.update({
                    "source_trial_idx": trial["trial_idx"],
                    "source_run_idx": trial["source_run_idx"],
                    "source_direction_idx": trial["direction_idx"],
                    "trial_timestamp": trial.get("timestamp", "unknown"),
                })
                
                if result["label"] == 1:
                    all_matching_results.append(enhanced_result)
                else:
                    all_non_matching_results.append(enhanced_result)
        
        # Calculate aggregated statistics
        all_results = all_matching_results + all_non_matching_results
        if not all_results:
            continue
            
        total_examples = len(all_results)
        correct_predictions = sum(1 for res in all_results if res["correct"])
        aggregated_accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
        
        # Calculate per-label accuracy
        matching_correct = sum(1 for res in all_matching_results if res["correct"])
        non_matching_correct = sum(1 for res in all_non_matching_results if res["correct"])
        matching_accuracy = matching_correct / len(all_matching_results) if all_matching_results else 0.0
        non_matching_accuracy = non_matching_correct / len(all_non_matching_results) if all_non_matching_results else 0.0
        
        # Trial-level statistics
        trial_accuracies = [trial["accuracy"] for trial in all_trials_for_direction 
                           if "accuracy" in trial]
        
        aggregated_direction = {
            "direction_idx": direction_idx,
            "explanation": direction_metadata["explanation"] if direction_metadata else "No explanation available",
            "css_score": direction_metadata["css_score"] if direction_metadata else 0.0,
            "num_source_runs": len(set(trial["source_run_idx"] for trial in all_trials_for_direction)),
            "num_trials": len(all_trials_for_direction),
            "total_examples": total_examples,
            "total_matching_examples": len(all_matching_results),
            "total_non_matching_examples": len(all_non_matching_results),
            "aggregated_accuracy": aggregated_accuracy,
            "matching_accuracy": matching_accuracy,
            "non_matching_accuracy": non_matching_accuracy,
            "correct_predictions": correct_predictions,
            "matching_correct": matching_correct,
            "non_matching_correct": non_matching_correct,
            "trial_mean_accuracy": np.mean(trial_accuracies) if trial_accuracies else 0.0,
            "trial_std_accuracy": np.std(trial_accuracies) if trial_accuracies else 0.0,
            "trial_accuracy_95_ci": _compute_confidence_interval(trial_accuracies, confidence=0.95) if trial_accuracies else (0.0, 0.0),
            "all_matching_results": all_matching_results,
            "all_non_matching_results": all_non_matching_results,
            "all_trials": all_trials_for_direction,
        }
        
        aggregated_directions.append(aggregated_direction)
    
    # Compute overall aggregated statistics
    total_aggregated_examples = sum(direction["total_examples"] for direction in aggregated_directions)
    total_aggregated_correct = sum(direction["correct_predictions"] for direction in aggregated_directions)
    overall_aggregated_accuracy = total_aggregated_correct / total_aggregated_examples if total_aggregated_examples > 0 else 0.0
    
    # Collect all trial accuracies for overall statistics
    all_trial_accuracies = []
    for direction in aggregated_directions:
        all_trial_accuracies.extend([trial["accuracy"] for trial in direction["all_trials"] 
                                   if "accuracy" in trial])
    
    overall_summary = {
        "num_directions": len(aggregated_directions),
        "num_source_runs": len(multi_run_results),
        "total_trials": len(all_trial_accuracies),
        "total_aggregated_examples": total_aggregated_examples,
        "total_aggregated_correct": total_aggregated_correct,
        "overall_aggregated_accuracy": overall_aggregated_accuracy,
        "overall_trial_mean_accuracy": np.mean(all_trial_accuracies) if all_trial_accuracies else 0.0,
        "overall_trial_std_accuracy": np.std(all_trial_accuracies) if all_trial_accuracies else 0.0,
        "overall_trial_accuracy_95_ci": _compute_confidence_interval(all_trial_accuracies, confidence=0.95) if all_trial_accuracies else (0.0, 0.0),
    }
    
    # Generate HTML visualization if output_dir is provided
    if output_dir:
        # Ensure output_dir exists first, then create html subdirectory for consolidated visualization files
        output_dir.mkdir(parents=True, exist_ok=True)
        html_dir = output_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)
        aggregated_html_file = html_dir / "faithfulness_aggregated.html"
        _generate_aggregated_faithfulness_html(
            aggregated_directions, overall_summary, str(aggregated_html_file),
            sort_by=sort_by, show_faithfulness_badges=True
        )
    
    return {
        "overall_summary": overall_summary,
        "aggregated_directions": aggregated_directions,
        "source_runs": multi_run_results,
    }


def sort_directions_by_faithfulness(
    direction_summaries: List[Dict], 
    sort_by: str = "mean_accuracy",
    ascending: bool = False
) -> List[Dict]:
    """
    Sort direction summaries by faithfulness metrics.
    
    Args:
        direction_summaries: List of direction summary dictionaries
        sort_by: Metric to sort by ("mean_accuracy", "css_score", "direction_idx", 
                "min_accuracy", "max_accuracy", "median_accuracy")
        ascending: If True, sort in ascending order; if False, descending
        
    Returns:
        Sorted list of direction summaries
    """
    def get_sort_key(direction):
        """Extract the sorting key from a direction summary."""
        if sort_by == "mean_accuracy":
            return direction.get("mean_accuracy", 0.0)
        elif sort_by == "css_score":
            return direction.get("css_score", 0.0)
        elif sort_by == "direction_idx":
            return direction.get("direction_idx", 0)
        elif sort_by == "min_accuracy":
            return direction.get("min_accuracy", 0.0)
        elif sort_by == "max_accuracy":
            return direction.get("max_accuracy", 0.0)
        elif sort_by == "median_accuracy":
            return direction.get("median_accuracy", 0.0)
        elif sort_by == "aggregated_accuracy":
            return direction.get("aggregated_accuracy", 0.0)
        else:
            # Default to mean_accuracy
            return direction.get("mean_accuracy", 0.0)
    
    return sorted(direction_summaries, key=get_sort_key, reverse=not ascending)


def sort_aggregated_directions_by_faithfulness(
    aggregated_directions: List[Dict],
    sort_by: str = "aggregated_accuracy", 
    ascending: bool = False
) -> List[Dict]:
    """
    Sort aggregated direction results by faithfulness metrics.
    
    Args:
        aggregated_directions: List of aggregated direction dictionaries
        sort_by: Metric to sort by ("aggregated_accuracy", "trial_mean_accuracy", 
                "css_score", "matching_accuracy", "non_matching_accuracy")
        ascending: If True, sort in ascending order; if False, descending
        
    Returns:
        Sorted list of aggregated directions
    """  
    def get_sort_key(direction):
        """Extract the sorting key from an aggregated direction."""
        if sort_by == "aggregated_accuracy":
            return direction.get("aggregated_accuracy", 0.0)
        elif sort_by == "trial_mean_accuracy":
            return direction.get("trial_mean_accuracy", 0.0)
        elif sort_by == "css_score":
            return direction.get("css_score", 0.0)
        elif sort_by == "matching_accuracy":
            return direction.get("matching_accuracy", 0.0)
        elif sort_by == "non_matching_accuracy":
            return direction.get("non_matching_accuracy", 0.0)
        elif sort_by == "direction_idx":
            return direction.get("direction_idx", 0)
        else:
            # Default to aggregated_accuracy
            return direction.get("aggregated_accuracy", 0.0)
    
    return sorted(aggregated_directions, key=get_sort_key, reverse=not ascending)

def load_faithfulness_results(file_path: str) -> Dict[str, Any]:
    """
    Load faithfulness results from full format.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Loaded results dictionary
    """
    import json
    import gzip
    from pathlib import Path
    
    file_path = Path(file_path)
    
    try:
        if file_path.suffix == '.gz':
            # Load gzipped file
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                results = json.load(f)
        else:
            # Load regular JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        print(f"✓ Results loaded from {file_path}")
        
        return results
        
    except Exception as e:
        print(f"Error: Failed to load results from {file_path}: {e}")
        raise


def _generate_aggregated_faithfulness_html(
    aggregated_directions: List[Dict], overall_summary: Dict, output_file: str,
    sort_by: str = "aggregated_accuracy", show_faithfulness_badges: bool = True
) -> None:
    """Generate HTML visualization for aggregated faithfulness testing results."""
    
    # Sort directions by faithfulness score (highest first by default)
    sorted_directions = sort_aggregated_directions_by_faithfulness(
        aggregated_directions, sort_by=sort_by, ascending=False
    )
    
    # Generate direction cards HTML
    direction_cards_html = ""
    for rank, direction in enumerate(sorted_directions, 1):
        direction_idx = direction["direction_idx"]
        explanation = direction["explanation"]
        css_score = direction["css_score"]
        aggregated_accuracy = direction["aggregated_accuracy"]
        total_examples = direction["total_examples"]
        total_matching = direction["total_matching_examples"]
        total_non_matching = direction["total_non_matching_examples"]
        correct_predictions = direction["correct_predictions"]
        matching_correct = direction["matching_correct"]
        non_matching_correct = direction["non_matching_correct"]
        matching_accuracy = direction["matching_accuracy"]
        non_matching_accuracy = direction["non_matching_accuracy"]
        num_source_runs = direction["num_source_runs"]
        num_trials = direction["num_trials"]
        
        # Build examples sections for matching and non-matching
        matching_examples_html = ""
        non_matching_examples_html = ""
        
        # Show a sample of matching results
        for i, result in enumerate(direction["all_matching_results"][:20]):  # Show first 20
            correct_class = "correct" if result["correct"] else "incorrect"
            activation = result["activation"]
            prediction = result["prediction"]
            source_run = result.get("source_run_idx", "?")
            source_trial = result.get("source_trial_idx", "?")
            
            matching_examples_html += f"""
                <div class="example {correct_class}">
                    <div class="example-text">"{result['text'][:150]}{'...' if len(result['text']) > 150 else ''}"</div>
                    <div class="example-details">
                        Prediction: {"Positive" if prediction else "Negative"} |
                        Activation: {activation:.3f} |
                        Run {source_run}, Trial {source_trial} |
                        {"✓ Correct" if result["correct"] else "✗ Incorrect"}
                    </div>
                </div>
            """
        
        # Show a sample of non-matching results  
        for i, result in enumerate(direction["all_non_matching_results"][:20]):  # Show first 20
            correct_class = "correct" if result["correct"] else "incorrect"
            activation = result["activation"]
            prediction = result["prediction"]
            source_run = result.get("source_run_idx", "?")
            source_trial = result.get("source_trial_idx", "?")
            
            non_matching_examples_html += f"""
                <div class="example {correct_class}">
                    <div class="example-text">"{result['text'][:150]}{'...' if len(result['text']) > 150 else ''}"</div>
                    <div class="example-details">
                        Prediction: {"Positive" if prediction else "Negative"} |
                        Activation: {activation:.3f} |
                        Run {source_run}, Trial {source_trial} |
                        {"✓ Correct" if result["correct"] else "✗ Incorrect"}
                    </div>
                </div>
            """
        
        if len(direction["all_matching_results"]) > 20:
            matching_examples_html += f"<div class='examples-note'>... and {len(direction['all_matching_results']) - 20} more matching examples</div>"
            
        if len(direction["all_non_matching_results"]) > 20:
            non_matching_examples_html += f"<div class='examples-note'>... and {len(direction['all_non_matching_results']) - 20} more non-matching examples</div>"
        
        # Determine faithfulness badge
        faithfulness_badge = ""
        faithfulness_class = ""
        if show_faithfulness_badges:
            if aggregated_accuracy >= 0.8:
                faithfulness_badge = f'<span class="faithfulness-badge high">High Faithfulness ({aggregated_accuracy:.0%})</span>'
                faithfulness_class = "high-faithfulness"
            elif aggregated_accuracy >= 0.6:
                faithfulness_badge = f'<span class="faithfulness-badge medium">Medium Faithfulness ({aggregated_accuracy:.0%})</span>'
                faithfulness_class = "medium-faithfulness"
            else:
                faithfulness_badge = f'<span class="faithfulness-badge low">Low Faithfulness ({aggregated_accuracy:.0%})</span>'
                faithfulness_class = "low-faithfulness"
        
        direction_cards_html += f"""
        <div class="direction-section {faithfulness_class}">
            <div class="direction-header" onclick="toggleDirection({direction_idx})">
                <div class="header-left">
                    <h2>#{rank} - Direction {direction_idx}</h2>
                    {faithfulness_badge}
                </div>
                <div class="direction-stats">
                    <span class="stat">Accuracy: {aggregated_accuracy:.1%}</span>
                    <span class="stat">Examples: {total_examples}</span>
                    <span class="stat">Runs: {num_source_runs}</span>
                </div>
                <span class="toggle-icon" id="toggle-{direction_idx}">▼</span>
            </div>
            <div class="direction-content" id="direction-{direction_idx}">
                <div class="direction-summary">
                    <div class="summary-stats">
                        <div class="stat-card">
                            <div class="stat-value">{aggregated_accuracy:.1%}</div>
                            <div class="stat-label">Aggregated Accuracy</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{correct_predictions}/{total_examples}</div>
                            <div class="stat-label">Correct Predictions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{matching_correct}/{total_matching}</div>
                            <div class="stat-label">Matching Accuracy</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{non_matching_correct}/{total_non_matching}</div>
                            <div class="stat-label">Non-Matching Accuracy</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{num_source_runs}</div>
                            <div class="stat-label">Source Runs</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{num_trials}</div>
                            <div class="stat-label">Total Trials</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{css_score:.4f}</div>
                            <div class="stat-label">CSS Score</div>
                        </div>
                    </div>
                    
                    <div class="explanation-box">
                        <h3 style="margin-top: 0; color: #1565c0;">Feature Explanation</h3>
                        <p>{explanation}</p>
                    </div>
                </div>
                
                <div class="examples-section">
                    <h3 style="color: #0056b3;">Aggregated Matching Examples ({total_matching} total)</h3>
                    <div class="example-section">
                        {matching_examples_html}
                    </div>
                    
                    <h3 style="color: #0056b3;">Aggregated Non-Matching Examples ({total_non_matching} total)</h3>
                    <div class="example-section">
                        {non_matching_examples_html}
                    </div>
                </div>
            </div>
        </div>
        """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aggregated Faithfulness Testing Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #212529;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        }}
        h1 {{
            color: #0056b3;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }}
        .overall-summary {{
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #2196f3;
        }}
        .overall-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border-left: 4px solid #0056b3;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #0056b3;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }}
        .direction-section {{
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            overflow: hidden;
        }}
        .direction-header {{
            background-color: #0056b3;
            color: white;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.3s;
        }}
        .direction-header:hover {{
            background-color: #004494;
        }}
        .direction-header h2 {{
            margin: 0;
            font-size: 1.5rem;
        }}
        .direction-stats {{
            display: flex;
            gap: 20px;
        }}
        .stat {{
            padding: 5px 10px;
            background-color: rgba(255,255,255,0.2);
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        .toggle-icon {{
            font-size: 1.2rem;
            transition: transform 0.3s;
        }}
        .toggle-icon.rotated {{
            transform: rotate(180deg);
        }}
        .direction-content {{
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .direction-content.hidden {{
            display: none;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .explanation-box {{
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }}
        .examples-section {{
            margin-top: 30px;
        }}
        .example-section {{
            margin: 20px 0;
        }}
        .example {{
            margin: 10px 0;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #28a745;
        }}
        .example.incorrect {{
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }}
        .example.correct {{
            background-color: #d4edda;
        }}
        .example-text {{
            font-weight: 500;
            margin-bottom: 8px;
        }}
        .example-details {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        .examples-note {{
            margin-top: 15px;
            font-style: italic;
            color: #6c757d;
            text-align: center;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .header-left {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }}
        .faithfulness-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .faithfulness-badge.high {{
            background-color: #28a745;
            color: white;
        }}
        .faithfulness-badge.medium {{
            background-color: #ffc107;
            color: #212529;
        }}
        .faithfulness-badge.low {{
            background-color: #dc3545;
            color: white;
        }}
        .direction-section.high-faithfulness .direction-header {{
            border-left: 6px solid #28a745;
        }}
        .direction-section.medium-faithfulness .direction-header {{
            border-left: 6px solid #ffc107;
        }}
        .direction-section.low-faithfulness .direction-header {{
            border-left: 6px solid #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Aggregated Faithfulness Testing Results</h1>
        
        <div class="overall-summary">
            <h2 style="margin-top: 0; color: #1565c0;">Overall Aggregated Statistics</h2>
            <div class="overall-stats">
                <div class="stat-card">
                    <div class="stat-value">{overall_summary['num_directions']}</div>
                    <div class="stat-label">Directions Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall_summary['num_source_runs']}</div>
                    <div class="stat-label">Source Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall_summary['total_trials']}</div>
                    <div class="stat-label">Total Trials</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall_summary['total_aggregated_examples']}</div>
                    <div class="stat-label">Total Examples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall_summary['overall_aggregated_accuracy']:.1%}</div>
                    <div class="stat-label">Aggregated Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overall_summary['overall_trial_mean_accuracy']:.1%}</div>
                    <div class="stat-label">Trial Mean Accuracy</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 15px; color: #6c757d;">
                This page shows aggregated results from {overall_summary['num_source_runs']} runs, 
                combining individual test examples into unified pools per direction.<br>
                <strong>Directions are sorted by {sort_by.replace('_', ' ')} (highest first)</strong>
            </div>
        </div>
        
        {direction_cards_html}
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 0.9em;">
            Generated by FourierFeatures Aggregated Faithfulness Testing
        </div>
    </div>
    
    <script>
        function toggleDirection(directionIdx) {{
            const content = document.getElementById('direction-' + directionIdx);
            const toggle = document.getElementById('toggle-' + directionIdx);
            
            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                toggle.classList.remove('rotated');
                toggle.textContent = '▼';
            }} else {{
                content.classList.add('hidden');
                toggle.classList.add('rotated');
                toggle.textContent = '▲';
            }}
        }}
        
        // Initialize all sections as expanded
        document.addEventListener('DOMContentLoaded', function() {{
            // All sections start expanded by default
        }});
    </script>
</body>
</html>
"""

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Success: Aggregated faithfulness HTML report saved to {output_file}")
    except Exception as e:
        print(f"Warning: Failed to write aggregated HTML file {output_file}: {e}")
