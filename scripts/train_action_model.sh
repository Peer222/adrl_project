#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J action_model
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_outputs/action-%j.out
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
python /bigwork/nhwpduep/adrl_project/action_model/train.py --seed 0 --latent_features 28 --lr_scheduler one_cycle_lr
