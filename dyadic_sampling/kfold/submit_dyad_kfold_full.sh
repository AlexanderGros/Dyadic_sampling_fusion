#!/bin/bash
# Submission script 
#SBATCH --job-name=kfold_full_2
#SBATCH --time=2-12:00:00 # d-hh:mm:ss  80h
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12288 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=kfold_full_2.out



#module purge
ml --quiet purge
ml --quiet releases/2020b

ml Keras/2.4.3-foss-2020b
ml matplotlib/3.3.3-foss-2020b
ml h5py/3.1.0-foss-2020b



echo "job start at $(date)"
python dyadic_all.py
echo "job end at $(date)"

