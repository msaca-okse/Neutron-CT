#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Segment
# -- choose queue --
# For gpu write gpuv100, gpua100, gpua10, gpua40. Check availability by bqueues -l gpua100, eg
#BSUB -q hpc
#BSUB -e error.log
#BSUB -o output.log
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 4:00 
# -- Number of cores requested -- 
#BSUB -n 16
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5

#python s2_watershed_filter.py
#python -c "from s3_watershed_eds import run_diffusion; run_diffusion()"
python -c "from s3_watershed_eds import place_mean_in_watersheds; place_mean_in_watersheds()"