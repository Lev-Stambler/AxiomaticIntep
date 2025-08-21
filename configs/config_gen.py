from typing import Union, List


def gen(
    model_name: str,
    model_name_sae: str,
    layer_starts: List[int],
    layer_end: int,
    token_offset: int,
    dict_size: int,
    layer_adds: bool = True,
    dataset="EleutherAI/the_pile_deduplicated",
    dataset_config: Union[None, str] = None,
    hook_type = "hook_resid_post" 
):

    s = "+" if layer_adds else ""
    dir_name = f"configs/{model_name.replace('/', '_')}-{token_offset}-offset-{s}{layer_end}-{dataset.replace('/', '_')}-{hook_type}"
    # If the directory doesn't exist, create it
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for layer_start in layer_starts:
        o = """# Configuration for our model

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Model to analyze (HuggingFace model name)
model_name = """ + f'"{model_name}"' + """
model_name_sae = """ + f'"{model_name_sae}"' + """

# Layer cutoff for analysis (which layer to analyze)
layer_cutoff = """ + str(layer_start) + """
# Random seed for reproducibility
seed = 69

target_norm = 1.0 # Set to 1

# Hook type for transformer-lens integration
hook_type = """ + f'"{hook_type}"' + """

# Maximum sequence length for processing
max_seq_len = 150
# Token offset for targeting (0 = current token, -1 = last token)
# Set to last token
target_token_offset = """ + str(token_offset) + """

# Whether to include output logits in intervention scoring
use_output_logits = false

target_layers = [""" + str((layer_start + layer_end) if layer_adds else layer_end) + """] 


# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Batch size for PGA (Projected Gradient Ascent) training
pga_batch_size = 16

# Batch size for evaluation
eval_batch_size = 128

# Number of PGA iterations
pga_its = 2_000

# Dictionary size for CSS
dict_size = """ + str(dict_size) + """

# Learning rate for optimization
learning_rate = 0.001

# Top-k parameter
k = 1

# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
eps = 0.0001
weight_decay = 0.01

# ============================================================================
# EXPERIMENT FLAGS
# ============================================================================

# Use Ax platform for hyperparameter optimization
use_ax = false
n_norm_discretization_steps = -1

# ============================================================================
# EARLY STOPPING PARAMETERS
# ============================================================================

# Enable early stopping during training
early_stopping_enabled = true

# Number of iterations to wait before stopping if no improvement
early_stopping_patience = 5

# Minimum change in loss to consider as improvement
early_stopping_min_delta = 0.001

# How often to evaluate for early stopping (every N iterations)
early_stopping_eval_freq = 10

# Early stopping parameters for Ax
early_stopping_patience_ax = 5
early_stopping_min_delta_ax = 0.001
early_stopping_eval_freq_ax = 50

# ============================================================================
# DATABASE AND SEARCH PARAMETERS
# ============================================================================

# Number of similar activations to find during search
num_similar_to_find = 20

# Size of the similarity search index
index_size = 25000 # 25,000

# Batch size for building the search index
index_batch_size = 32

# Similarity scoring type for activation search: "dot" (inner product) or "cosine" (cosine similarity)
scoring_type = "cosine"

# ============================================================================
# DATASET PARAMETERS
# ============================================================================


dataset_name = """ + f'"{dataset}"' + """
""" + (f'dataset_config = "{dataset_config}"' if dataset_config != None else "") + """

# Whether to use streaming mode (recommended for large datasets)
dataset_streaming = true

# Name of the text column in the dataset
dataset_text_column = "text"

# ============================================================================
# DATASET CACHING PARAMETERS
# ============================================================================

# Force re-download even if cache exists
force_dataset_download = false

explainer_llm_provider = "openai"
explainer_llm_model_name = "o4-mini"
faithfulness_llm_provider = "openai"
faithfulness_llm_model_name = "o4-mini"

"""
        with open(f"{dir_name}/layer_{layer_start}.toml", "w") as f:
            f.write(o)
    print("Done with config gen")


if __name__ == "__main__":
    DICT_SIZE = 20

    def gen_pythia():
        
        # gen(
        #     model_name="EleutherAI/pythia-70m-deduped",
        #     model_name_sae="pythia-70m-deduped-mlp-sm",
        #     layer_starts=list(range(0, 5)),
        #     layer_end=1,  # Target the output layer
        #     layer_adds=True,
        #     token_offset=0,
        #     dict_size=DICT_SIZE,
        #     hook_type="hook_mlp_out"
        # )
        # gen(
        #     model_name="EleutherAI/pythia-70m-deduped",
        #     model_name_sae="pythia-70m-deduped-mlp-sm",
        #     layer_starts=list(range(0, 5)),
        #     layer_end=1,  # Target the output layer
        #     layer_adds=True,
        #     token_offset=1,
        #     dict_size=DICT_SIZE,
        #     hook_type="hook_mlp_out"
        # )
        gen(
            model_name="EleutherAI/pythia-70m-deduped",
            model_name_sae="pythia-70m-deduped-res-sm",
            layer_starts=list(range(0, 5)),
            layer_end=5,  # Target the output layer
            layer_adds=False,
            token_offset=0,
            dict_size=DICT_SIZE,
        )
        gen(
            model_name="EleutherAI/pythia-70m-deduped",
            model_name_sae="pythia-70m-deduped-res-sm",
            layer_starts=list(range(0, 5)),
            layer_end=1,
            layer_adds=True,
            token_offset=0,
            dict_size=DICT_SIZE,
        )
        gen(
            model_name="EleutherAI/pythia-70m-deduped",
            model_name_sae="pythia-70m-deduped-res-sm",
            layer_starts=list(range(0, 4)),
            layer_end=2,
            layer_adds=True,
            token_offset=1,
            dict_size=DICT_SIZE,
        )
        gen(
            model_name="EleutherAI/pythia-70m-deduped",
            model_name_sae="pythia-70m-deduped-res-sm",
            layer_starts=list(range(0, 5)),
            layer_end=1,
            layer_adds=True,
            token_offset=1,
            dict_size=DICT_SIZE,
        )

    gen_pythia()
