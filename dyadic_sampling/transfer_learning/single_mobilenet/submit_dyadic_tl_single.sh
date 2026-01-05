#!/bin/bash
# Submission script 
#SBATCH --job-name=gpu_tl_single
#SBATCH --time=0-00:01:00 # d-hh:mm:ss  80h
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12 
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=24576 # 120GB   36864  #  8192 12288  24576!  36864  49152
#SBATCH --partition=gpu
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=gpu_tl_singlet.out



#module purge
ml --quiet purge
ml --quiet releases/2023a
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1



echo "job start at $(date)"
python dyadic_tl_single.py
echo "job end at $(date)"