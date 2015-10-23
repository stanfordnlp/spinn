#!/bin/bash

### Generic job script for all experiments.
# TODO(SB): Figure out how/if this interacts with Theano GPU use.

#PBS -l nodes=1:ppn=4 	### Request at least 6 cores
#PBS -l walltime=99:00:00	### Die after four days
#PBS -l mem=4000MB
#PBS -q nlp

# Usage example:
# export REMBED_FLAGS="--learning_rate 0.2"; qsub -v REMBED_FLAGS run.sh

# Change to the submission directory.
cd $PBS_O_WORKDIR
echo Lauching from working directory: $PBS_O_WORKDIR
echo Flags: $REMBED_FLAGS

# Log what we're running and where.
echo `hostname` - $PBS_JOBID - $REMBED_FLAGS >> ~/rembed_machine_assignments.txt

python -m rembed.models.classifier $REMBED_FLAGS
