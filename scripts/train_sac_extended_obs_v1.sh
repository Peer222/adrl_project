#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J sac_door
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_outputs/extended_obs-%j.out
#SBATCH --partition=tnt,ai
#SBATCH --time=6:00:00

#SBATCH --mail-user=p.duensing@stud.uni-hannover.de
#SBATCH --mail-type=FAIL,END
cd $SLURM_SUBMIT_DIR

module load Miniconda3

source /home/nhwpduep/.bashrc

conda activate /bigwork/nhwpduep/.conda/envs/adrl_project

export WANDB_MODE=offline

# change seed
python /bigwork/nhwpduep/adrl_project/algorithms/sac_ca_extended_obs.py --action_model_dir "action_runs/AdroitHandDoor-v1_action_model_s0_2024-09-03_14:17:00.053531/models" --seed 0
