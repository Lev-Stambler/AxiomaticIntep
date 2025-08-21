import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import copy

from sae_lens import SAE
from interventionfeatures.core import css
import torch
import time

try:
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except ImportError:
    print("TK Inter not installed, assuming no GUI")
# Optional LangChain imports for explanation generation
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableLambda

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    raise e

# Local imports
from ..core.search import ActivationSimSearcher
from ..core.model import IntervenableTransformerSegment
from ..core.data_handler import TransformerDataHandler



prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
you are an expert in mechanistic interpretability. I will provide text examples where a specific
feature in a language model has high activation. the activations appear in parentheses after each token.""" + \
"""\n your task is to provide a single, concise, abstract explanation of the concept this feature represents based on the highlighted tokens (highlighted by two stars, **) and the context surrounding the token
(consider the rest of the context as well being part of the explanation. I.e. if the samples are all different but the same language or describing some similar context like science, code, etc., feel free to use that context as a description)
            """,
                ),
                (
                    "human",
                    "here are the top activating examples with activations following each token in parentheses:\n{examples}\n\n.",
                ),
            ]
        )



# A simple debug utility to inspect data in a LangChain pipe
def debug_print(data, label=""):
    """Prints data with a label for debugging purposes within a chain."""
    print(f"--- DEBUG: {label} ---")
    print(data)
    print("--- END DEBUG ---")
    return data


#def robust_json_extract(text: str) -> Optional[Dict]:
#    """Extracts JSON from text that may contain additional content before/after JSON."""
#    try:
#        # First try parsing the entire text as JSON
#        return json.loads(text.strip())
#    except json.JSONDecodeError:
#        pass
#
#    # Look for JSON-like content using regex
#    json_pattern = r'\{[^{}]*"explanation"[^{}]*\}'
#    matches = re.findall(json_pattern, text, re.DOTALL)
#
#    for match in matches:
#        try:
#            return json.loads(match)
#        except json.JSONDecodeError:
#            continue
#
#    # Try to find content between curly braces
#    brace_pattern = r"\{.*\}"
#    matches = re.findall(brace_pattern, text, re.DOTALL)
#
#    for match in matches:
#        try:
#            return json.loads(match)
#        except json.JSONDecodeError:
#            continue
#
#    return None
#

def validate_tensor_shapes(
    css_direction: torch.Tensor, scores_by_vec: torch.Tensor, sample_idx: int, searcher
) -> bool:
    """Validates tensor shapes and indices for safe access."""
    try:
        if scores_by_vec.dim() < 2:
            print(
                f"Warning: Scores tensor should be at least 2D, got {scores_by_vec.dim()}D"
            )
            return False

        if not hasattr(searcher, "indexed_sample_tokens"):
            print("Warning: Searcher missing indexed_sample_tokens attribute")
            return False

        if sample_idx >= len(searcher.indexed_sample_tokens):
            print(f"Warning: Sample index {sample_idx} out of bounds")
            return False

        return True
    except Exception as e:
        print(f"Warning: Tensor validation failed: {e}")
        return False


class FeatureExplainer:
    """
    Handles the generation of textual explanations for neural network features
    based on their top activating examples.

    Uses counterfactual tests (projecting onto zero space) 
    """

    def __init__(
        self,
        model: IntervenableTransformerSegment,
        explainer_llm_provider: str = "anthropic",
        explainer_llm_model_name: str = "claude-3-opus-20240229",
        temperature: float = 0.0,
    ):
        """
        Initializes the FeatureExplainer.

        Args:
            model: The intervenable transformer segment.
            explainer_llm_provider: The language model provider for explanation generation (e.g., 'anthropic', 'openai').
            explainer_llm_model_name: The model name to use for explanation generation.
            temperature: The temperature to use for the model's output.
        """
        self.model = model
        self.llm_provider = explainer_llm_provider
        self.llm_model_name = explainer_llm_model_name
        self.temperature = temperature

        # Check for LangChain availability
        if not LANGCHAIN_AVAILABLE:
            print(
                "Warning: LangChain not available. Install with: pip install langchain langchain-anthropic langchain-openai"
            )
            self.langchain_available = False
        else:
            self.langchain_available = True

        # Check for API key availability
        self.api_available = self._check_api_key()

        if self.api_available:
            try:
                self.explanation_chain = self._create_explanation_chain()
            except Exception as e:
                print(f"wError: Failed to initialize explanation chains: {e}")
                self.api_available = False
                raise e

    def _check_api_key(self) -> bool:
        """Checks if the required API key for the selected provider is available."""
        if self.llm_provider == "anthropic":
            api_key_env = "ANTHROPIC_API_KEY"
        elif self.llm_provider == "openai":
            api_key_env = "OPENAI_API_KEY"
        else:
            print(f"Warning: Unknown LLM provider '{self.llm_provider}'. No API key check.")
            return False

        if not os.environ.get(api_key_env):
            print(
                f"Warning: {api_key_env} not found. Explanation features will be limited."
            )
            return False
        return True

    def _get_chat_model(self):
        """Returns the chat model instance based on the provider."""
        if self.llm_provider == "anthropic":
            if ChatAnthropic is None:
                raise ImportError("langchain-anthropic is not installed.")
            return ChatAnthropic(model=self.llm_model_name, temperature=self.temperature)
        elif self.llm_provider == "openai":
            if ChatOpenAI is None:
                raise ImportError("langchain-openai is not installed.")
            return ChatOpenAI(model=self.llm_model_name)#, temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")


    def _create_explanation_chain(self) -> Runnable:
        """creates a langchain chain to generate explanations for feature behavior."""
        model = self._get_chat_model()
        parser = StrOutputParser()

        return prompt_template | model | parser


    def get_top_activating_examples(
        self,
        css_direction: torch.Tensor,
        searcher: ActivationSimSearcher,
        data_handler: TransformerDataHandler,
        num_examples: int = 5,
        shift_with_relu : Optional[float] = None
    ) -> Tuple[List[str], List[torch.Tensor], List[int], float]:
        """
        Gets top activating text examples for a CSS direction.

        This method finds examples and formats them with activation scores.

        Args:
            css_direction: The feature direction vector.
            searcher: An initialized ActivationSimSearcher instance.
            data_handler: An initialized TransformerDataHandler instance.
            num_examples: The number of examples to format and return.

        Returns:
            A tuple containing a list of formatted text examples and the max activation score.
        """
        top_k = num_examples
        print(f"Searching for top {top_k} activating examples...")
        similar_results, scores_by_vec = searcher.search_with_aggregation(
            css_direction,
            top_k=top_k,
            aggregation="mean",  # Using mean can sometimes be more stable than min/max
            rerank_top_n=max(10 * top_k, 1000),
            use_cosine_scoring=False
        )

        if not similar_results:
            print("Warning: No similar results found by the searcher.")
            return [], 0.0

        # Normalize scores by the highest score found across all results for better comparison
        max_score = similar_results[0]["score"]
        if max_score > 0 and shift_with_relu is None:
            scores_by_vec /= max_score
        else:
            print("Warning: Max score is 0, skipping normalization")

        top_examples_list = []
        original_toks = []
        top_toks_inds = []

        print(f"Formatting top {num_examples} examples...")
        for example_idx, result in enumerate(similar_results[:num_examples]):
            sample_idx = result["sample_idx"]

            # Validate tensor shapes and indices
            if not validate_tensor_shapes(
                css_direction, scores_by_vec, sample_idx, searcher
            ):
                print(
                    f"Warning: Skipping example {example_idx} due to validation failure"
                )
                continue

            sample_tokens = searcher.indexed_sample_tokens[sample_idx]

            # Safely access the tensor with bounds checking
            try:
                if example_idx >= scores_by_vec.shape[0]:
                    print(
                        f"Warning: example_idx {example_idx} out of bounds for scores tensor"
                    )
                    continue

                sample_scores_tensor = scores_by_vec[example_idx, 0, :]
                top_token_idx = torch.argmax(sample_scores_tensor).item()

                # Ensure we don't access beyond the token sequence length
                max_tokens = min(len(sample_tokens), sample_scores_tensor.shape[0])
                if max_tokens == 0:
                    print(f"Warning: No valid tokens for example {example_idx}")
                    continue

                # Truncate to valid length
                sample_tokens = sample_tokens[:max_tokens]
                sample_scores_tensor = sample_scores_tensor[:max_tokens]

                # Find the token with the highest activation in this specific example
            except Exception as e:
                raise ValueError(f"Warning: Error processing example {example_idx}: {e}")
            top_toks_inds.append(top_token_idx)
            original_toks.append(sample_tokens)

            formatted_tokens = []
            for token_idx, token_id in enumerate(sample_tokens):
                try:
                    # Handle potential tokenizer decode failures
                    token_text = data_handler.tokenizer.decode([token_id])
                    if not token_text:  # Handle empty decode results
                        token_text = f"[UNK_{token_id}]"
                except Exception as decode_error:
                    print(
                        f"Warning: Token decode failed for token_id {token_id}: {decode_error}"
                    )
                    token_text = f"[ERR_{token_id}]"

                try:
                    token_score = sample_scores_tensor[token_idx].item()
                except Exception as score_error:
                    print(
                        f"Warning: Score access failed for token {token_idx}: {score_error}"
                    )
                    raise f"Score access failed for token {token_idx}: {score_error}"

                if shift_with_relu is not None:
                    t = token_score + shift_with_relu 
                    token_score = t if t > 0.0 else 0.0
                # Format the token with its score
                formatted_token = f"{token_text}({token_score:.3f})"

                # Add emphasis to the highest-activating token
                if token_idx == top_token_idx:
                    formatted_token = f"**{formatted_token}**"

                formatted_tokens.append(formatted_token)

            # Join the tokens to form the full, untruncated text
            full_text = "".join(formatted_tokens)

            top_examples_list.append(full_text)

        assert len(top_examples_list) == len(top_toks_inds)

        return top_examples_list, original_toks, top_toks_inds, max_score

    def generate_explanation(self, examples: List[str],
                                            ) -> str:
        """
        Generates an explanation for a feature based on its top activating examples.

        Args:
            examples: A list of formatted text examples.
            next_tokens: A list of next tokens without intervention
            next_tokens: A list of next tokens with intervention of removing the direction

        Returns:
            An explanation string, or an error message if generation fails.
        """
        if not examples:
            return "No examples were provided, cannot generate explanation."

        if not self.api_available:
            return "API unavailable - basic pattern detected from examples"

        formatted_examples = "\n\n".join(f"- {ex} " for i, ex in enumerate(examples))
        
        print("FORMATTED EXAMPLES!", formatted_examples)
        

        print("Generating explanation with the language model...")
        MAX_RETRIES = 5
        for i in range(MAX_RETRIES):
            try:
                # The invoke call will execute the chain: prompt -> model -> parser
                explanation_data = self.explanation_chain.invoke(
                    {"examples": formatted_examples}
                )
                explanation_text = explanation_data
                print(f"Generated Explanation: {explanation_text}")
                return explanation_text
            except Exception as e:
                sleep_time = 2 ** (i + 1)
                print(f"explanation generation/ parsing failed, attempting retry after {sleep_time} seconds. Error: {e}")
                time.sleep(sleep_time)
                if i == MAX_RETRIES - 1:
                    raise e
            
    def explain_css_direction(
        self,
        css_direction: torch.Tensor,
        searcher: ActivationSimSearcher,
        data_handler: TransformerDataHandler,
        num_examples: int = 10,
        shift_with_relu : Optional[float] = None
    ) -> Tuple[str, List[str], float]:
        """
        Performs the full pipeline of explaining a single feature direction.

        Args:
            css_direction: The feature direction vector.
            searcher: An initialized ActivationSimSearcher instance.
            data_handler: An initialized TransformerDataHandler instance.
            num_examples: The number of examples to format and return.

        Returns:
            A tuple containing the explanation, the list of examples used, and the max activation.
        """
        assert css_direction.dim() == 1 or (css_direction.dim() ==
                                            2 and css_direction.shape[0] == 1), "Expected a single direction"

        # 1. Get top activating examples
        top_examples, original_tokens, top_tok_inds, max_act = self.get_top_activating_examples(
            css_direction=css_direction,
            searcher=searcher,
            data_handler=data_handler,
            num_examples=num_examples,
            shift_with_relu = shift_with_relu
        )
        print("ZZZZ", list(zip([len(o) for o in original_tokens], top_tok_inds)))

        # Squeeze for remainder
        if css_direction.dim() == 2:
            css_direction = css_direction.squeeze(0)

        self.model.base_model.eval()  # Ensure model is in eval mode
        explanation = self.generate_explanation(top_examples)

        return explanation, top_examples, max_act

    def explain_css_directions(
        self,
        css_directions: List[Dict[str, torch.Tensor]],
        searcher: ActivationSimSearcher,
        data_handler: TransformerDataHandler,
        num_examples: int = 10,
    ) -> List[Tuple[str, List[str], float]]:
        """
        Runs the explanation pipeline for a list of CSS directions.
        This is the method called by Main2.py.

        Args:
            css_directions: A list of dictionaries, each containing a tensor 's'.
            searcher: An initialized ActivationSimSearcher instance.
            data_handler: An initialized TransformerDataHandler instance.
            num_examples: The number of examples to format and return.

        Returns:
            A list of tuples, where each tuple contains (explanation, examples, max_activation).
        """
        results = []
        for _, direction_dict in enumerate(css_directions):
            # print(f"\n--- Explaining Direction {i+1}/{len(css_directions)} ---")
            direction_tensor = direction_dict["s"]
            explanation_result = self.explain_css_direction(
                direction_tensor,
                searcher,
                data_handler,
                num_examples,
            )
            results.append(explanation_result)
        return results

    def explain_css_directions_positive_negative(
        self,
        css_directions: List[Dict[str, torch.Tensor]],
        searcher: ActivationSimSearcher,
        data_handler: TransformerDataHandler,
        num_examples: int = 10,
    ) -> List[Dict[str, Tuple[str, List[str], float]]]:
        """
        Runs the explanation pipeline for a list of CSS directions, generating both
        positive and negative explanations.

        Args:
            css_directions: A list of dictionaries, each containing a tensor 's'.
            searcher: An initialized ActivationSimSearcher instance.
            data_handler: An initialized TransformerDataHandler instance.
            num_examples: The number of examples to format and return.

        Returns:
            A list of dictionaries, where each dictionary contains:
            {
                "positive": (explanation, examples, max_activation),
                "negative": (explanation, examples, max_activation)
            }
        """
        results = []
        for i, direction_dict in enumerate(css_directions):
            print(f"\n--- Explaining Direction {i+1}/{len(css_directions)} (Positive & Negative) ---")
            direction_tensor = direction_dict["s"]
            
            # Generate positive explanation (original direction)
            print(f"Generating positive explanation for direction {i+1}...")
            positive_result = self.explain_css_direction(
                direction_tensor,
                searcher,
                data_handler,
                num_examples,
            )
            
            # Generate negative explanation (negated direction)
            print(f"Generating negative explanation for direction {i+1}...")
            negative_direction = -direction_tensor
            negative_result = self.explain_css_direction(
                negative_direction,
                searcher,
                data_handler,
                num_examples,
            )
            
            results.append({
                "positive": positive_result,
                "negative": negative_result
            })
        return results

    def explain_sae_feature(
        self,
        sae: SAE,
        feature_idx: int,
        searcher: ActivationSimSearcher,
        data_handler: TransformerDataHandler,
        num_examples: int = 10,
    ) -> Tuple[str, List[str], float]:
        """
        Explain an SAE feature using the current pipeline
        """
        print(f"Searching for top {num_examples} SAE feature {feature_idx} activating examples from indexed dataset...")
        
        # Calculate SAE feature activations for all samples in the searcher's dataset
        print("Computing SAE activations for all indexed samples...")
        sae_dir: torch.Tensor = sae.W_enc[:, feature_idx].detach().unsqueeze(0)
        # SAEs for Pythia shift by the inner product the decoding offset on the input
        sae_cutoff = sae.b_enc[feature_idx].detach() - torch.inner(sae.b_dec, sae_dir.squeeze()).detach()

        # Get all the indexed activations from the searcher
        # These are the model activations at the hook point that were pre-computed
        if not searcher.is_indexed:
            raise AttributeError("ActivationSimSearcher must have indexed_activations for SAE analysis")

        return self.explain_css_direction(
            sae_dir,
            searcher,
            data_handler,
            shift_with_relu=sae_cutoff # Shift by 
        )
