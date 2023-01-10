#!/bin/bash

export PYTHONPATH=src

for track in configurations/eval_run_cfgs/veealign_datasets/*; do
	echo "Running Evaluations for Track $(basename $track) ..."
	for ali_pair in $track/*; do
		echo "Alignment Pair $(basename $ali_pair) ..."
		for cfg in $ali_pair/*.yaml; do
			python src/scripts/run_oa_eval.py -rc "$cfg"
		done
		python src/scripts/generate_merged_evaluation_report.py data/eval_results/veealign_datasets/$(basename $track)/$(basename $ali_pair)
	done
	python src/scripts/generate_merged_evaluation_report.py data/eval_results/veealign_datasets/$(basename $track)
done

