#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100
#BSUB -q hpc
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 24000
#BSUB -R "rusage[mem=24000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 1:00 
# -- Number of cores requested -- 
#BSUB -n 12
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]" 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


for ALPHA in "0.1" "0.2" "0.5" "0.75"
do

# Export variables to make them available to the Python script
export ALPHA

python preprocessor_XA.py


done

