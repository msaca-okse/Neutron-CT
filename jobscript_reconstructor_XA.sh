#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100
#BSUB -q gpua40
#BSUB -e my_job_error2.log
#BSUB -o my_job_output2.log
#BSUB -M 2000
#BSUB -R "rusage[mem=2000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
# -- Number of cores requested -- 
#BSUB -n 16
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
#BSUB -J folder_job[37-100]
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


ALPHA="150"
NUM_PROC="100"
STRIDE="1"
N_ITER="120"

# Export variables to make them available to the Python script
export ALPHA
export NUM_PROC
export STRIDE
export N_ITER

# python -c "from reconstructor_XA import recon_FBP_single; recon_FBP_single($LSB_JOBINDEX)"
# python -c "from reconstructor_XA import recon_FBP_multi; recon_FBP_multi($LSB_JOBINDEX)"
# python -c "from reconstructor_XA import recon_TV_single; recon_TV_single($LSB_JOBINDEX)"
python -c "from reconstructor_XA import recon_TV_multi; recon_TV_multi($LSB_JOBINDEX)"


# Estimated completion time for 100 iterations for the whole volume is 48 hours





