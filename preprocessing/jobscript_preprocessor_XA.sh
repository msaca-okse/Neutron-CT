#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100
#BSUB -q hpc
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 20000
#BSUB -R "rusage[mem=20000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 10:00 
# -- Number of cores requested -- 
#BSUB -n 10
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]" 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5

STRIDE="1"
export STRIDE

ALPHA="0.2"
export ALPHA

python preprocessor_XA.py


# ALPHA = ca. 0.2
