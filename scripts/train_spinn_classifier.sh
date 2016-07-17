#!/bin/bash

### Generic job script for all experiments.

# Usage example:
# export SPINN_FLAGS="--learning_rate 0.01 --batch_size 256"; export DEVICE=gpu2; export DEVICE=gpu0; qsub -v SPINN_FLAGS,DEVICE scripts/train_spinn_classifier.sh -l host=jagupard10

# Change to the submission directory.
cd $PBS_O_WORKDIR
echo Lauching from working directory: $PBS_O_WORKDIR
echo Flags: $SPINN_FLAGS
echo Device: $DEVICE

# Log what we're running and where.

# Use Jon's Theano install.
source /u/nlp/packages/anaconda/bin/activate conda-common
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/scr/jgauthie/tmp/theano-nshrdlu:$PYTHONPATH
export THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu0,floatX=float32 

echo $PBS_JOBID - `hostname` - at `git log --pretty=format:'%h' -n 1` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

stake.py -g $MEM "python -m spinn.models.classifier $SPINN_FLAGS"
