#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J sac_baseline
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_outputs/baseline-%j.out
#SBATCH --partition=tnt,ai
#SBATCH --time=24:00:00

#SBATCH --mail-user=p.duensing@stud.uni-hannover.de
#SBATCH --mail-type=FAIL,END
cd $SLURM_SUBMIT_DIR

module load Miniconda3

source /home/nhwpduep/.bashrc

conda activate /bigwork/nhwpduep/.conda/envs/adrl_project

export WANDB_MODE=offline

# change seed
python /bigwork/nhwpduep/adrl_project/algorithms/sac_ca_baseline.py --seed 1 --exp_name door_baseline_1 --wandb_project_name adrl_results
python /bigwork/nhwpduep/adrl_project/algorithms/sac_ca_baseline.py --seed 2 --exp_name door_baseline_2 --wandb_project_name adrl_results
python /bigwork/nhwpduep/adrl_project/algorithms/sac_ca_baseline.py --seed 3 --exp_name door_baseline_3 --wandb_project_name adrl_results
python /bigwork/nhwpduep/adrl_project/algorithms/sac_ca_baseline.py --seed 4 --exp_name door_baseline_4 --wandb_project_name adrl_results
python /bigwork/nhwpduep/adrl_project/algorithms/sac_ca_baseline.py --seed 5 --exp_name door_baseline_5 --wandb_project_name adrl_results
