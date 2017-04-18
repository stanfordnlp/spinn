#!/bin/bash

# Generic job script for all experiments.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -t24:00:00

#PRINCE PRINCE_GPU_COMPUTE_MODE=default

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 pytorch/0.1.11_5 protobuf/intel/3.1.0

# Run.
export IFS=';'
for SUB_FLAGS in $SPINN_FLAGS
do
	unset IFS
	python -m spinn.models.fat_classifier  --noshow_progress_bar --gpu 0 $SUB_FLAGS &
done

wait 
