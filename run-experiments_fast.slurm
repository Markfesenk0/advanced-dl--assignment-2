#! /bin/sh

#SBATCH --job-name=fast
#SBATCH --output=logs/fast.out # redirect stdout
#SBATCH --error=logs/fast.err # redirect stderr
##SBATCH --account=gpu-research
#SBATCH --partition=studentkillable
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=20G # CPU memory (MB)
#SBATCH --gpus=1


python train.py --sampler_type FastDPM
