#!/bin/bash
# Submission script 
#SBATCH --job-name=dsf_test
#SBATCH --time=0-00:05:00 # d-hh:mm:ss 860h     3-08
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24576 # 120GB   36864  #  8192 12288  24576  36864  49152
#SBATCH --partition=batch
#
#SBATCH --mail-user=alexander.gros@umons.ac.be
#SBATCH --mail-type=ALL
#SBATCH --output=dsf_test.out



#module purge
ml --quiet purge
ml --quiet releases/2020b

ml Keras/2.4.3-foss-2020b
ml matplotlib/3.3.3-foss-2020b
ml h5py/3.1.0-foss-2020b



echo "job start at $(date)"
python dyadic_s_fusion_5.py
echo "job end at $(date)"

