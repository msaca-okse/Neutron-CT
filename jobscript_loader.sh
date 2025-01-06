#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Registration
# -- choose queue --
#BSUB -q hpc
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 128000
#BSUB -R "rusage[mem=128000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
# -- Number of cores requested -- 
#BSUB -n 1
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"

# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5
python loader_fullXA_NA.py
