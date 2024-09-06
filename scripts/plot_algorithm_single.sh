#!/bin/bash
module load Miniconda3
# source "/${home}/.bashrc"

conda activate adrl_project

for experiment in "baseline Baseline" "extended_obs Extended_Observations" "latent_actions Latent_Actions" "latent_actions_extended Latent_Actions_with_Extended_Observations"
do
    set -- $experiment
    python utils/single_plots.py --result_dir "plots/${1}" --file_paths "experiment_data/${1}" --run_labels $2
done
