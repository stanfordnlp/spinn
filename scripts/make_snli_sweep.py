# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "ENC"
SWEEP_RUNS = 2

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "/home/sb6065/snli_1.0/snli_1.0_train.jsonl", "")
gflags.DEFINE_string("eval_data_path", "/home/sb6065/snli_1.0/snli_1.0_dev.jsonl", "")
gflags.DEFINE_string("embedding_data_path", "/home/sb6065/glove/glove.840B.300d.txt", "")
gflags.DEFINE_string("log_path", "/home/sb6065/logs", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "snli",
    "model_type":      "RLSPINN",
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
    "statistics_interval_steps": "1000",
    "use_internal_parser": "",
    "batch_size":  "64",
    "use_encode": "",
    "encode_reverse": "",
    "noencode_bidirectional": "",
    "num_mlp_layers": "2",
    "transition_weight": "1.0",
    "rl_entropy": "",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      ("lr", EXP, 0.0002, 0.002),  # RNN likes higher, but below 009.
    "l2_lambda":          ("l2", EXP, 8e-7, 2e-5),
    "semantic_classifier_keep_rate": ("skr", LIN, 0.7, 0.95),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": ("ekr", LIN, 0.7, 0.95),
    "learning_rate_decay_per_10k_steps": ("dec", EXP, 0.5, 1.0),
    "tracking_lstm_hidden_dim": ("tdim", EXP, 24, 128),
    "rl_weight":  ("rlwt", EXP, 0.00001, 0.01),
    "rl_entropy_beta": ("rle", EXP, 0.00001, 0.1)
}

sweep_name = "sweep_" + NAME + "_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(SWEEP_RUNS)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(SWEEP_RUNS):
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
        flags += " --" + param + " " + str(value)

    flags += " --experiment_name " + name
    if NYU_NON_PBS:
        print "cd spinn/python; python2.7 -m spinn.models.fat_classifier " + flags
    else:
        print "SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit.sh"
    print
