#!/bin/bash
module load Miniconda3
# source "/${home}/.bashrc"

conda activate adrl_project

python utils/single_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/total_losses_train.csv" "experiment_data/action-model/contrastive_losses_train.csv" "experiment_data/action-model/reconstruction_losses_train.csv" --smoothing_window 1000 --step_resolution 1000 --run_labels Training Training Training
python utils/single_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/total_losses_val.csv" "experiment_data/action-model/contrastive_losses_val.csv" "experiment_data/action-model/reconstruction_losses_val.csv" --smoothing_window 0 --step_resolution 0 --run_labels Validation Validation Validation

for loss in "total_losses" "reconstruction_losses" "contrastive_losses";
do
    python utils/multi_plots.py --result_dir "plots/action-model" --file_paths "experiment_data/action-model/${loss}_train.csv" "experiment_data/action-model/${loss}_val.csv" --smoothing_window 1000 --step_resolution 1000 --run_labels Training Validation
    mv "plots/action-model/multi_${loss}_train.png" "plots/action-model/multi_${loss}.png"
done