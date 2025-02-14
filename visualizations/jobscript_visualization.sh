#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100
#BSUB -q gpuv100
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 10000
#BSUB -R "rusage[mem=10000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 4:00 
# -- Number of cores requested -- 
#BSUB -n 2
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]" 


source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5

module load VirtualGL
vglrun vtk_example.py



