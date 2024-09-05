#!/bin/bash
module load Miniconda3
source /home/nhwpduep/.bashrc

conda activate /bigwork/nhwpduep/.conda/envs/adrl_project

wandb_entity=$1
wandb_project=$2

if [ -z $1 ] || [ -z $2 ];
then 
    echo "No parameter passed. Make sure to log in to your wandb account with 'wandb login' and pass your wandb entity as first and your created wandb project as second parameter"
else
    python get_algorithm_runs.py --result_dir experiment_data/baseline --wandb_runs "${wandb_entity}/${wandb_project}/door_baseline_1" "${wandb_entity}/${wandb_project}/door_baseline_2" "${wandb_entity}/${wandb_project}/door_baseline_3" "${wandb_entity}/${wandb_project}/door_baseline_4" "${wandb_entity}/${wandb_project}/door_baseline_5"
fi
