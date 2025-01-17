#!/bin/bash

# Submit the first job and capture its job ID
job1_id=$(bsub < jobscript_preprocessor_XA.sh | awk '{print $2}' | tr -d '<>')

# Submit the second job with dependency on the first job
job2_id=$(bsub -w "done(${job1_id})" < jobscript_reconstructor_XA.sh | awk '{print $2}' | tr -d '<>')

# Submit the third job with dependency on the second job
job3_id=$(bsub -w "done(${job2_id})" < jobscript_cleaner_XA.sh | awk '{print $2}' | tr -d '<>')

