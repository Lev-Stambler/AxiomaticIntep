CF=$1

echo "Running config file $CF"

source .env
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main sae-faithfulness --config-file configs/EleutherAI_pythia-70m-deduped-0-offset-+1-EleutherAI_the_pile_deduplicated-hook_resid_post/layer_0.toml
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main sae-faithfulness --config-file configs/EleutherAI_pythia-70m-deduped-0-offset-+1-EleutherAI_the_pile_deduplicated-hook_resid_post/layer_1.toml
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main sae-faithfulness --config-file configs/EleutherAI_pythia-70m-deduped-0-offset-+1-EleutherAI_the_pile_deduplicated-hook_resid_post/layer_2.toml
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main sae-faithfulness --config-file configs/EleutherAI_pythia-70m-deduped-0-offset-+1-EleutherAI_the_pile_deduplicated-hook_resid_post/layer_3.toml
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main sae-faithfulness --config-file configs/EleutherAI_pythia-70m-deduped-0-offset-+1-EleutherAI_the_pile_deduplicated-hook_resid_post/layer_4.toml
