#!/bin/bash

### Generic job script for all experiments.

# Usage example:
# export REMBED_FLAGS="--learning_rate 0.2"; qsub -v REMBED_FLAGS run.sh

### The below are not comments, but actual PBS flags:
#PBS -l walltime=99:00:00	### Die after four days
#PBS -l mem=4000MB   ### Request 4G RAM
#PBS -q jag   ### Run on the jagupard* queue

# Change to the submission directory.
cd $PBS_O_WORKDIR
echo Lauching from working directory: $PBS_O_WORKDIR
echo Flags: $REMBED_FLAGS
DEVICE=`python pick_gpu.py`

# Log what we're running and where.
echo `hostname` - $PBS_JOBID - $REMBED_FLAGS - $DEVICE - at `git log --pretty=format:'%h' -n 1` >> ~/rembed_machine_assignments.txt

THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=$DEVICE,floatX=float32 python -m rembed.models.classifier $REMBED_FLAGS
