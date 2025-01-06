#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
#BSUB -q gpuv100
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 8000
#BSUB -R "rusage[mem=8000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 3:00 
# -- Number of cores requested -- 
#BSUB -n 16
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"

# -- end of LSF options -- 
export NUM_PROCS=$LSB_DJOB_NUMPROC

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


THRESHOLD="0.5"
SIZE="7"
DECNUM="4"
WNAME="5"
SIGMA="0.1"

# Export variables to make them available to the Python script
export THRESHOLD
export SIZE
export DECNUM
export WNAME
export SIGMA
python reconstructor_NA_fbp.py


: <<'EOF'

PARAM1=0.5
PARAM2=5
python reconstructor_NA_fbp.py --param1 "$PARAM1" --param2 "$PARAM2"

PARAM1=0.5
PARAM2=10
python reconstructor_NA_fbp.py --param1 "$PARAM1" --param2 "$PARAM2"

PARAM1=0.5
PARAM2=15
python reconstructor_NA_fbp.py --param1 "$PARAM1" --param2 "$PARAM2"

PARAM1=0.5
PARAM2=25
python reconstructor_NA_fbp.py --param1 "$PARAM1" --param2 "$PARAM2"

PARAM1=0.5
PARAM2=35
python reconstructor_NA_fbp.py --param1 "$PARAM1" --param2 "$PARAM2"

PARAM1=0.5
PARAM2=50
python reconstructor_NA_fbp.py --param1 "$PARAM1" --param2 "$PARAM2"
EOF
