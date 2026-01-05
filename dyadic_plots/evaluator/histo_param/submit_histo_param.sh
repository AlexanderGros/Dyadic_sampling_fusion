#!/bin/bash
# Submission script 
#SBATCH --job-name=histo_param
#SBATCH --time=0-00:15:00 # d-hh:mm:ss  60h
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8192 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=histo_param.out



#module purge

ml --quiet purge
ml --quiet releases/2023a
ml Python
ml matplotlib/3.7.2-gfbf-2023a



#ml --quiet purge
#ml --quiet releases/2020b
#ml Keras/2.4.3-foss-2020b
#ml matplotlib/3.3.3-foss-2020b
#ml h5py/3.1.0-foss-2020b



echo "job start at $(date)"
python histo_param.py
echo "job end at $(date)"

