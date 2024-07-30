#! /bin/sh

#SBATCH --job-name=ddpm
#SBATCH --output=logs/ddpm.out # redirect stdout
#SBATCH --error=logs/ddpm.err # redirect stderr
#SBATCH --partition=killable
#SBATCH --time=1300 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time's up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=20G # CPU memory (MB)
#SBATCH --gpus=1


python train.py
