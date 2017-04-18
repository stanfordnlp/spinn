#!/bin/bash

# Sets up a series of 10 jobs to run one after the other to get around time limits.
# Optionally specify an sbatch script to use.

bat=${1:-../scripts/train_spinn.sbatch}
num_runs=${2:-7}

jobID=
for((i=0; i<$num_runs; i++)); do
    if [ "$jobID" == "" ]; then
        jobID=$(sbatch -o ~/logs/slurm-%j.out -e ~/logs/slurm-%j.err $bat | awk '{print $NF}')
    else
        jobID=$(sbatch -o ~/logs/slurm-%j.out -e ~/logs/slurm-%j.err --dependency=afterany:$jobID $bat | awk '{print $NF}')
    fi
done
