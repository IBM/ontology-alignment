#!/bin/bash

export PYTHONPATH=src

for cfg in configurations/eval_run_cfgs/oaei_anatomy/*.yaml; do
	python src/scripts/run_oa_eval.py -rc "$cfg"
done

python src/scripts/generate_merged_evaluation_report.py data/eval_results/oaei_anatomy

