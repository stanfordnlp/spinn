# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np

LIN = "LIN"
EXP = "EXP"

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
    "model_dim":   "200",
    "seq_length":	"50",
    "eval_seq_length":	"100",
    "clipping_max_value":  "5.0",
    "batch_size":  "32",
    "lstm_composition": "",
    "ckpt_root":    os.path.join("/afs/cs.stanford.edu/u", getpass.getuser(), "scr/")  # Launching user's home scr dir
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      (EXP, 0.00005, 0.001),
    "l2_lambda":   		  (EXP, 5e-7, 1e-4),
    "init_range":         (EXP, 0.001, 0.005),
    "double_identity_init_range": (EXP, 0.0005, 0.005),
    "semantic_classifier_keep_rate": (LIN, 0.5, 0.9),
    "embedding_keep_rate": (LIN, 0.5, 1.0)
}

sweep_name = "sweep_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]
sweep_runs = 6
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
    print "export REMBED_FLAGS=\"" + flags + "\"; export DEVICE=gpuX; qsub -v REMBED_FLAGS,DEVICE train_rembed_classifier.sh -q " + queue + " -l host=jagupardX"
    print
