#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -e my_job_error1.log
#BSUB -o my_job_output1.log
#BSUB -M 6000
#BSUB -R "rusage[mem=6000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 2:00 
# -- Number of cores requested -- 
#BSUB -n 10
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
#BSUB -J folder_job[1-10]
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


ALPHA="0.75"
NUM_PROC="10"
STRIDE="10"

# Export variables to make them available to the Python script
export ALPHA
export NUM_PROC

#python -c "from reconstructor_NA import recon_FBP_single; recon_FBP_single($LSB_JOBINDEX)"
python -c "from reconstructor_NA import recon_FBP_multi; recon_FBP_multi($LSB_JOBINDEX)"
# python -c "from reconstructor_NA import recon_TV_single; recon_TV_single($LSB_JOBINDEX)"
# python -c "from reconstructor_NA import recon_TV_multi; recon_TV_multi($LSB_JOBINDEX)"







