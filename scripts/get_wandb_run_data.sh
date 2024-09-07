#!/bin/bash
module load Miniconda3
# source "/${home}/.bashrc"

conda activate adrl_project

wandb_entity=$1
wandb_project=$2

if [ -z $1 ] || [ -z $2 ];
then 
    echo "No parameter passed. Make sure to log in to your wandb account with 'wandb login' and pass your wandb entity as first and your created wandb project as second parameter"
else
    for experiment in "baseline" "extended_obs" "latent_actions_obs_only" "latent_actions_extended";
    do
        python utils/get_algorithm_runs.py --result_dir "experiment_data/${experiment}" \
            --wandb_runs "${wandb_entity}/${wandb_project}/door_${experiment}_1" \
                "${wandb_entity}/${wandb_project}/door_${experiment}_2" \
                "${wandb_entity}/${wandb_project}/door_${experiment}_3" \
                "${wandb_entity}/${wandb_project}/door_${experiment}_4" \
                "${wandb_entity}/${wandb_project}/door_${experiment}_5"
    done
fi
