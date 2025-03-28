#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100, gpua100, gpua10, gpua40. Check availability by bqueues -l gpua100, eg
#BSUB -q gpua100
#       #BSUB -gpu "num=1"
#BSUB -e error.log
#BSUB -o output.log
#BSUB -M 1000
#BSUB -R "rusage[mem=1000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 15:00 
# -- Number of cores requested -- 
#BSUB -n 16
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
#BSUB -J dfrctn_prep
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5
module load cuda/12.2
python reconstructor_DA_2d.py

