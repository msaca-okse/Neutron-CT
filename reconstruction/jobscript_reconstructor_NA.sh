#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100, gpua100, gpua10, gpua40. Check availability by bqueues -l gpua100, eg
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -e error.log
#BSUB -o output.log
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 10:00 
# -- Number of cores requested -- 
#BSUB -n 8
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
#BSUB -J folder_job[1-4]
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5


# python -c "from reconstructor_NA import recon_FBP_single; recon_FBP_single($LSB_JOBINDEX, num_proc = 4)"
# python -c "from reconstructor_NA import recon_FBP_multi; recon_FBP_multi($LSB_JOBINDEX, num_proc = 4, output = False)"
# python -c "from reconstructor_NA import recon_TV_single; recon_TV_single($LSB_JOBINDEX, num_proc = 4, N_iter = 50, alpha = 75.0)"
# python -c "from reconstructor_NA import recon_TV_multi; recon_TV_multi($LSB_JOBINDEX, num_proc = 4, N_iter = 50, alpha = 75.0)"
python -c "from reconstructor_NA import recon_dTV_multi; recon_dTV_multi($LSB_JOBINDEX, num_proc = 4, N_iter = 50, alpha = 75.0, eta = 0.002)"

