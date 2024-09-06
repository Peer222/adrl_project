#!/bin/bash
module load Miniconda3
# source "/${home}/.bashrc"

conda activate adrl_project

python single_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/total_losses_train.csv" "experiment_data/action-model/contrastive_losses_train.csv" "experiment_data/action-model/reconstruction_losses_train.csv" --smoothing_window 1000 --step_resolution 1000 --run_labels Training Training Training
python single_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/total_losses_val.csv" "experiment_data/action-model/contrastive_losses_val.csv" "experiment_data/action-model/reconstruction_losses_val.csv" --smoothing_window 0 --step_resolution 0 --run_labels Validation Validation Validation

python multi_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/total_losses_train.csv" "experiment_data/action-model/total_losses_val.csv" --smoothing_window 1000 --step_resolution 1000 --run_labels Training Validation
python multi_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/reconstruction_losses_train.csv" "experiment_data/action-model/reconstruction_losses_val.csv" --smoothing_window 1000 --step_resolution 1000 --run_labels Training Validation
python multi_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/contrastive_losses_train.csv" "experiment_data/action-model/contrastive_losses_val.csv" --smoothing_window 1000 --step_resolution 1000 --run_labels Training Validation
