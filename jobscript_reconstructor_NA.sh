#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100, gpua100, gpua10, gpua40. Check availability by bqueues -l gpua100, eg
#BSUB -q gpua40
#BSUB -gpu "num=1"
#BSUB -e error.log
#BSUB -o output.log
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 10:00 
# -- Number of cores requested -- 
#BSUB -n 12
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
#BSUB -J folder_job[1-10]
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


THRESHOLD="0.5"
SIZE="7"
DECNUM="5"
WNAME="10"
SIGMA="0.3"
ALPHA="75.0"
NUM_PROC="10"
BETA="1"

# Export variables to make them available to the Python script
export THRESHOLD
export SIZE
export DECNUM
export WNAME
export SIGMA
export ALPHA
export BETA
export DELTA
export NUM_PROC

# python -c "from reconstructor_NA import recon_FBP_single; recon_FBP_single($LSB_JOBINDEX)"
# python -c "from reconstructor_NA import recon_FBP_multi; recon_FBP_multi($LSB_JOBINDEX)"
# python -c "from reconstructor_NA import recon_TV_single; recon_TV_single($LSB_JOBINDEX)"
python -c "from reconstructor_NA import recon_TV_multi; recon_TV_multi($LSB_JOBINDEX)"

