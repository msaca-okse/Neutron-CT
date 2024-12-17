#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J DataCompression
# -- choose queue --
#BSUB -q hpc
# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job.
#BSUB -R "rusage[mem=1GB]"
# -- Notify me by email when execution begins --
# -- Notify me by email when execution ends   --
# -- email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
# -- Output File --
#BSUB -o ./joblogs/Output_%J.out
# -- Error File --
#BSUB -e ./joblogs/Error_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 2:00 
# -- Number of cores requested -- 
#BSUB -n 7
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: 7 tasks, one per folder
#BSUB -J folder_job[1-7]

# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5
python -c "from sliceA_data_compressor import folder_processor; folder_processor($LSB_JOBINDEX)"

