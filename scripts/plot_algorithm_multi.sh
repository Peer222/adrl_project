#!/bin/bash
module load Miniconda3
# source "/${home}/.bashrc"

conda activate adrl_project
for metric in "episodic_returns" "actor_losses" "qf_losses"
do
    file_paths=""
    run_labels=""
    for experiment in "baseline Baseline" "extended_obs Extended_Observations" "latent_actions Latent_Actions" "latent_actions_extended Latent_Actions_with_Extended_Observations"
    do
        set -- $experiment
        file_paths="${file_paths} experiment_data/${1}/${metric}.csv"
        run_labels="${run_labels} ${2}"
    done
    python multi_plots.py --result_dir "plots/" --file_paths $file_paths --run_labels $run_labels
done
