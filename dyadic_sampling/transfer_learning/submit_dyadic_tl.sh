#!/bin/bash
# Submission script 
#SBATCH --job-name=TL_mn2_df_4096
#SBATCH --time=3-08:00:00 # d-hh:mm:ss  60h
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12          
#SBATCH --mem-per-cpu=36864 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=TL_mn2_df_4096.out



#module purge
ml --quiet purge
ml --quiet releases/2023a
ml Python
ml matplotlib/3.7.2-gfbf-2023a




echo "job start at $(date)"
python dyadic_tl.py
echo "job end at $(date)"