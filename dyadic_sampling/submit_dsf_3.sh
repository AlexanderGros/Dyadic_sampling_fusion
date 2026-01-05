#!/bin/bash
# Submission script 
#SBATCH --job-name=dsf_512_higher_2
#SBATCH --time=0-00:10:00 # d-hh:mm:ss  60h
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24576 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=dsf_512_higher_2.out



#module purge
ml --quiet purge
ml --quiet releases/2023a
ml Python
ml matplotlib/3.7.2-gfbf-2023a



echo "job start at $(date)"
python dyadic_s_fusion_3.py
echo "job end at $(date)"

