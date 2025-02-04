#!/bin/bash

#BSUB -q hpc          # Specify the queue
#BSUB -e error.log  # Top-level error log
#BSUB -o output.log # Top-level output log
#BSUB -M 100           # Memory (16 GB)
#BSUB -W 24:00           # Wall time (24 hours)
#BSUB -n 1              # Number of CPU cores
#BSUB -R "span[hosts=1]" # Run on a single host
#BSUB -J "Submitter"

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5

# Define the list of folders (1 to 7)
folders=(1 2 3 4 5 6 7)

# Define the number of batches per folder
# Change this to the desired number of batches


N_batches=4
# Loop over each folder
for folder in "${folders[@]}"; do
    bsub -q gpua40 \
         -e error.log \
         -o output.log \
         -M 12000 \
         -R "rusage[mem=12000]" \
         -W 8:00 \
         -n 8 \
         -R "span[hosts=1]" \
         -J "folder_job_${folder}[1-${N_batches}]" <<EOF

        # Inside the job script: access the folder and batch index
        FOLDER=${folder}
	N_batches=4
	export N_batches        
       	export FOLDER
	export BATCH_ID=\$LSB_JOBINDEX
        # Start GPU monitoring
	echo "Starting GPU usage monitoring..."
	nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits --loop-ms=1000 > gpu_usage.log &
	MONITOR_PID=\$!
        # Perform the processing for the current folder and batch
        echo "Processing folder \$folder, batch \$batch_index"
	source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil5
        ALPHA="120"
	BETA="1"
	STRIDE="1"
	N_ITER="50"

	export ALPHA
	export BETA
	export STRIDE
	export N_ITER
	export LSB_JOBINDEX
	module load cuda


	# python -c "from reconstructor_XA import recon_FBP_single; recon_FBP_single($LSB_JOBINDEX)"
        # python -c "from reconstructor_XA import recon_FBP_multi; recon_FBP_multi($LSB_JOBINDEX)"
        # python -c "from reconstructor_XA import recon_TV_single; recon_TV_single($LSB_JOBINDEX)"
        python -c "from reconstructor_XA import recon_TV_multi; recon_TV_multi()"
	kill \$MONITOR_PID
	echo "GPU monitoring stopped. Check gpu_usage.log for details."

        # Add your actual batch processing logic here
        # For example, you could call a script to process the folder and batch:
        # ./process_folder.sh \$folder \$batch_index
EOF
done

# -- end of LSF options -- 


# Estimated completion time for 100 iterations for the whole volume is 48 hours
