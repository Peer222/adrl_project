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
    python utils/get_action_model_runs.py --result_dir experiment_data/action-model \ 
        --wandb_runs "${wandb_entity}/${wandb_project}/door_action-model_1" \ 
            "${wandb_entity}/${wandb_project}/door_action-model_2" \ 
            "${wandb_entity}/${wandb_project}/door_action-model_3" \ 
            "${wandb_entity}/${wandb_project}/door_action-model_4" \ 
            "${wandb_entity}/${wandb_project}/door_action-model_5"
fi
