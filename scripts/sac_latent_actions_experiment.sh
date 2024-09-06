#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J sac_latent_actions_obs_only
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_outputs/latent_actions_obs_only-%j.out
#SBATCH --partition=tnt,ai
#SBATCH --time=24:00:00

cd $SLURM_SUBMIT_DIR

module load Miniconda3
# source "/${home}/.bashrc"

conda activate adrl_project

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ];
then 
    echo "No parameter passed. Make sure to log in to your wandb account with 'wandb login' and pass the following arguments:
    1. your wandb entity,
    2. your created wandb project,
    3. wandb mode [online, offline] (if offline, manual upload with wandb sync ... is needed after training)"
else
    wandb_entity=$1
    wandb_project=$2
    export WANDB_MODE=$3

    for i in $(seq 1 5);
    do
        python algorithms/sac_ca_latent_actions.py --action_model_dir "action_runs/door_action-model_${i}/models" --observation_input obs_only --seed $i --exp_name "door_latent_actions_obs_only_${i}" --wandb_entity $wandb_entity --wandb_project_name $wandb_project
    done
fi
