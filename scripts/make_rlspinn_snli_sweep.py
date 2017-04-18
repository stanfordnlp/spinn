"""
Create a script to run a random hyperparameter search.

Example Usage:

python scripts/make_rlspinn_snli_sweep.py  \
--sweep_path ec2_02_16 \
--sweep_id ec2 \
--rl_baseline policy \
--rl_reward xent \
--sweep_runs 2 \
--gpu -1 \
--ec2 \
--embedding_data_path /efs/ec2-user/glove.840B.300d.txt \
--training_data_path /efs/ec2-user/snli_1.0/snli_1.0_train.jsonl \
--eval_data_path /efs/ec2-user/snli_1.0/snli_1.0_dev.jsonl \
--log_path /efs/ec2-user/logs

"""

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"

FLAGS = gflags.FLAGS

# Experiment path settings
gflags.DEFINE_string("training_data_path", "/scratch/apd283/snli_1.0/snli_1.0_train.jsonl", "")
gflags.DEFINE_string("eval_data_path", "/scratch/apd283/snli_1.0/snli_1.0_dev.jsonl", "")
gflags.DEFINE_string("embedding_data_path", "/scratch/apd283/glove/glove.840B.300d.txt", "")
gflags.DEFINE_string("log_path", "/scratch/apd283/shared-logs", "")

gflags.DEFINE_enum("rl_baseline", "ema", ["ema", "policy", "greedy"], "")
gflags.DEFINE_enum("rl_reward", "standard", ["standard", "xent"], "")

# Sweep settings
gflags.DEFINE_string("sweep_path", "./sweep", "")
gflags.DEFINE_string("sweep_id", "", "")
gflags.DEFINE_integer("sweep_runs", 4, "")

gflags.DEFINE_integer("gpu", -1, "")
gflags.DEFINE_boolean("ec2", False, "")

FLAGS(sys.argv)

print(FLAGS.FlagValuesDict())

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "snli",
    "model_type":      "RLSPINN",
    "rl_baseline":      FLAGS.rl_baseline,
    "rl_reward":      FLAGS.rl_reward,
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "embedding_data_path": FLAGS.embedding_data_path,
    "log_path": FLAGS.log_path,
    "metrics_path": FLAGS.log_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "300",
    "model_dim":   "600",
    "seq_length":   "150",
    "eval_seq_length":  "150",
    "eval_interval_steps": "1000",
    "statistics_interval_steps": "500",
    "use_internal_parser": "",
    "batch_size":  "32",
    "use_encode": "",
    "gpu": str(FLAGS.gpu),
    "encode_bidirectional": "",
    "num_mlp_layers": "2",
    "training_steps": "250001",
    "noshow_progress_bar": "",
    "lowercase": "",
    "transition_weight": "1.0",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      ("lr", EXP, 0.00001, 0.002),
    "l2_lambda":          ("l2", EXP, 8e-7, 2e-5),
    "semantic_classifier_keep_rate": ("skr", LIN, 0.7, 0.95),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": ("ekr", LIN, 0.7, 0.95),
    "learning_rate_decay_per_10k_steps": ("dec", EXP, 0.5, 1.0),
    "tracking_lstm_hidden_dim": ("tdim", EXP, 24, 128),
    "rl_weight":  ("rlwt", EXP, 0.00001, 0.1),
}

sweep_name = "sweep_" + FLAGS.sweep_id + "_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"] + \
    "_" + FIXED_PARAMETERS["rl_baseline"] + "_" + FIXED_PARAMETERS["rl_reward"]

GPU_template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=/scratch/apd283/shared-slurm/slurm_%j.out

module load pytorch/intel/20170125

pip install --user python-gflags==2.0

export PYTHONPATH=$PYTHONPATH:../python:./python

cd /scratch/apd283/shared-dev/spinn/checkpoints

{command}
"""

CPU_template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --output=/scratch/apd283/shared-slurm/slurm_%j.out

module load pytorch/intel/20170125

pip install --user python-gflags==2.0

export PYTHONPATH=$PYTHONPATH:../python:./python

cd /scratch/apd283/shared-dev/spinn/checkpoints

{command}
"""

ec2_template = """#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../python:./python
cd /efs/ec2-user/spinn/checkpoints

{command}
"""

tpl = GPU_template if FLAGS.gpu >= 0 else CPU_template
tpl = ec2_template if FLAGS.ec2 else tpl

# - #

for run_id in range(FLAGS.sweep_runs):
    params = {}
    name = sweep_name + "_" + str(run_id)

    params.update(FIXED_PARAMETERS)
    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[1]
        mn = config[2]
        mx = config[3]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        elif t==SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
        else: 
            val_disp = "%.2g" % sample

        params[param] = sample
        name += "-" + config[0] + val_disp

    flags = ""
    for param in params:
        value = params[param]
        val_str = ""
        flags += " \\\n --" + param + " " + str(value)

    flags += " \\\n --experiment_name " + name

    command = "python2.7 -m spinn.models.fat_classifier " + flags

    open(os.path.join(FLAGS.sweep_path, name + ".sh"), 'w').write(tpl.format(command=command))
