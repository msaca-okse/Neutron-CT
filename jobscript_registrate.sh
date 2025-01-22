#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Registration
# -- choose queue --
#BSUB -q hpc
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 32000
#BSUB -R "rusage[mem=32000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
# -- Number of cores requested -- 
#BSUB -n 4
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"

# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5
python registrate_XA_NA.py
