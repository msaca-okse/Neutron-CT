#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
#BSUB -q gpuv100
#BSUB -e my_job_error1.log
#BSUB -o my_job_output1.log
#BSUB -M 10000
#BSUB -R "rusage[mem=10000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 3:00 
# -- Number of cores requested -- 
#BSUB -n 12
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"

# -- end of LSF options -- 
export NUM_PROCS=$LSB_DJOB_NUMPROC

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


THRESHOLD="0.5"
SIZE="7"
DECNUM="5"
WNAME="10"
SIGMA="0.3"
ALPHA="50.0"

# Export variables to make them available to the Python script
export THRESHOLD
export SIZE
export DECNUM
export WNAME
export SIGMA
export ALPHA
export BETA
export DELTA

python reconstructor_NA_fbp.py

 
# BETA="100"
# export BETA
# python reconstructor_NA_fbp.py

# ALPHA="5.0"
# export ALPHA
# python reconstructor_NA_fbp.p

# DECNUM="7"
# export DECNUM
# python reconstructor_NA_fbp.py



