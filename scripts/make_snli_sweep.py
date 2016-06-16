# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "snli",
    "model_type":     "Model0",
    "training_data_path":    "/scr/nlp/data/snli_1.0/snli_1.0_train.jsonl",
    "eval_data_path":    "/scr/nlp/data/snli_1.0/snli_1.0_dev.jsonl",
    "embedding_data_path": "/scr/nlp/data/glove_vecs/glove.840B.300d.txt",
    "word_embedding_dim":	"300",
    "model_dim":   "600",
    "seq_length":	"50",
    "eval_seq_length":	"50",
    "batch_size":  "32",
    "ckpt_path":    os.path.join("/scr/", getpass.getuser(), "/"),  # Launching user's home scr dir
    "log_path":    os.path.join("/scr/", getpass.getuser(),  "/")  # Launching user's home scr dir
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      (EXP, 0.0002, 0.01),  # RNN likes higher end of range, but below 009.
    "l2_lambda":   		  (EXP, 8e-7, 2e-5),
    "semantic_classifier_keep_rate": (LIN, 0.80, 0.95),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": (LIN, 0.8, 0.95),
    "scheduled_sampling_exponent_base": (SS_BASE, 1e-5, 8e-5),
    "transition_cost_scale": (LIN, 0.5, 4.0),
    "tracking_lstm_hidden_dim": (EXP, 24, 128),
    "num_sentence_pair_combination_layers": (LIN, 1, 3)
}

sweep_name = "sweep_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]
sweep_runs = 4
queue = "jag"

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(sweep_runs)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(sweep_runs):
    params = {}
    params.update(FIXED_PARAMETERS)
    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[0]
        mn = config[1]
        mx = config[2]

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

        params[param] = sample

    name = sweep_name + "_" + str(run_id)
    flags = ""
    for param in params:
        value = params[param]
        val_str = ""
        flags += " --" + param + " " + str(value)
        if param not in FIXED_PARAMETERS:
            if isinstance(value, int):
                val_disp = str(value)
            else:
                val_disp = "%.2g" % value
            name += "-" + param + val_disp
    flags += " --experiment_name " + name
    print "export SPINN_FLAGS=\"" + flags + "\"; export DEVICE=gpuX; qsub -v SPINN_FLAGS,DEVICE ../scripts/train_spinn_classifier.sh -q " + queue + " -l host=jagupardX"
    print
