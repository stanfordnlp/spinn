# Create a script to run a random hyperparameter search.

import copy
import random
import numpy as np

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

sweep_name = "test_sweep"
sweep_runs = 12
queue = "nlp"

# Tunable parameters.
SWEEP_PARAMETERS = {
    "seq_length":    (LIN, 25, 100),
    "embedding_dim":    (EXP, 10, 200),
    "learning_rate":    (EXP, 0.001, 0.1),
    "momentum":     (LIN, 0., 0.999),
    "clipping_max_norm":    (LIN, 0.5, 10.0),
    "l2_lambda":    (EXP, 1e-7, 1e-3),
    "init_range":    (EXP, 0.01, 0.2)
}

# Non-tunable flags that must be passed in.
FIXED_PARAMETERS = {
    "data_type":     "sst",
    "training_data_path":    "sst-data/train.txt",
    "eval_data_path":    "sst-data/dev.txt"
}

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

    name = sweep_name + str(run_id)
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
    print "export REMBED_FLAGS=\"" + flags + "\"; qsub -v REMBED_FLAGS run.sh -q " + queue
    print
