CF=$1

echo "Running config file $CF"

source .env
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main find-directions --config-file $CF && \
echo "DONE WITH DIRECTION FINDING \n\n" && \
echo "" && \
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main explain-display --config-file $CF && \
echo "DONE WITH DISPLAY EXPLAIN \n\n" && \
echo "" && \
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main generate-vis --config-file $CF && \
PYTHONPATH=$(pwd)/src/ python3 -m interventionfeatures.cli.main validate --sort-by-faithfulness css_score --config-file $CF && \
echo "DONE WITH DISPLAY EXPLAIN \n\n" && \
echo ""
