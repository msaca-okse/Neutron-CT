#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpua40
#BSUB -q gpua10
# #BSUB -gpu "num=1"
#BSUB -e my_job_error.log
#BSUB -o my_job_output.log
#BSUB -M 4000
#BSUB -R "rusage[mem=4000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 14:00 
# -- Number of cores requested -- 
#BSUB -n 12
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]" 

# source ~/.bashrc
export PATH="/zhome/71/c/146676/miniconda3/bin:$PATH" && source /zhome/71/c/146676/miniconda3/etc/profile.d/conda.sh && conda activate
conda activate cil5

module load cuda/12.2
#module load cudnn/v9.8.0.87-prod-cuda-12.X
#module load hdf5/1.1
nvidia-smi
python preprocessor_DA.py



