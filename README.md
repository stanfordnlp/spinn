*NOTE:* This codebase is under active development. To exactly reproduce the experiments published in ACL 2016, use [this release][7].

# Stack-augmented Parser-Interpreter Neural Network

This repository contains the source code described in our paper [A Fast Unified Model for Sentence Parsing and Understanding][1]. There are three separate implementations available:

- A **Python/Theano** implementation of SPINN using a naïve stack representation (named `fat-stack`)
- A **Python/Theano** implementation of SPINN using the `thin-stack` representation described in our paper
- A **C++/CUDA** implementation of the SPINN feedforward, used for performance testing

## Python code

The Python code lives, quite intuitively, in the `python` folder. We used this code to train and test the SPINN models before publication.

There is one enormous difference in the `fat-` and `thin-stack` implementations: `fat-stack` uses Theano's automatically generated symbolic backpropagation graphs, while `thin-stack` generates its own optimal backpropagation graph. This makes `thin-stack` oodles faster than its brother, but we have not yet implemented all SPINN variants to support this custom backpropagation.

### Installation

Requirements:

- Python 2.7
- CUDA >= 7.0
- CuDNN == v4 (v5 is not compatible with our Theano fork)

Install all required Python dependencies using the command below. (**WARNING:** This installs our custom Theano fork. We recommend installing in a virtual environment in order to avoid overwriting any stock Theano install you already have.)

    pip install -r python/requirements.txt

We use [a modified version of Theano][3] in order to support fast forward- and backward-prop in `thin-stack`. While it isn't absolutely necessary to use this hacked Theano, it greatly improves `thin-stack` performance.

Alternatively, you can use a custom Docker image that we've prepared, as discussed in this [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0xa85b2da5365f423d952f800370ebb9b5/).

### Running the code

The easiest way to launch a train/test run is to use one of the [`checkpoints` directory](https://github.com/stanfordnlp/spinn/tree/master/checkpoints).
The Bash scripts in this directory will download the necessary data and launch train/test runs of all models reported in our paper. You can run any of the following scripts:

    ./checkpoints/spinn.sh
    ./checkpoints/spinn_pi.sh
    ./checkpoints/spinn_pi_nt.sh
    ./checkpoints/rnn.sh

All of the above scripts will by default launch a training run beginning with the recorded parameters of our best models. You can change their behavior using the arguments below:

    $ ./checkpoints/spinn.sh -h
    spinn.sh [-h] [-e] [-t] [-s] -- run a train or test run of a SPINN model

    where:
        -h    show this help text
        -e    run in eval-only mode (evaluates on dev set by default)
        -t    evaluate on test set
        -s    skip the checkpoint loading; run with a randomly initialized model

To evaluate our best SPINN-PI-NT model on the test set, for example, run

    $ ./checkpoints/spinn_pi_nt.sh -e -t
    Running command:
    python -m spinn.models.fat_classifier --data_type snli --embedding_data_path ../glove/glove.840B.300d.txt --log_path ../logs --training_data_path ../snli_1.0/snli_1.0_train.jsonl --experiment_name spinn_pi_nt --expanded_eval_only --eval_data_path ../snli_1.0/snli_1.0_test.jsonl --ckpt_path spinn_pi_nt.ckpt_best   --batch_size 32 --embedding_keep_rate 0.828528124124 --eval_seq_length 50 --init_range 0.005 --l2_lambda 3.45058959758e-06 --learning_rate 0.000297682444894 --model_dim 600 --model_type Model0 --noconnect_tracking_comp  --num_sentence_pair_combination_layers 2 --semantic_classifier_keep_rate 0.9437038157 --seq_length 50 --tracking_lstm_hidden_dim 57 --use_tracking_lstm  --word_embedding_dim 300
    ...
    [1] Checkpointed model was trained for 156500 steps.
    [1] Building forward pass.
    [1] Writing eval output for ../snli_1.0/snli_1.0_test.jsonl.
    [1] Written gold parses in spinn_pi_nt-snli_1.0_test.jsonl-parse.gld
    [1] Written predicted parses in spinn_pi_nt-snli_1.0_test.jsonl-parse.tst
    [1] Step: 156500    Eval acc: 0.808734   0.000000   ../snli_1.0/snli_1.0_test.jsonl

#### Custom model configurations

The main executable for the SNLI experiments in the paper is [fat_classifier.py](https://github.com/stanfordnlp/spinn/blob/master/python/spinn/models/fat_classifier.py), whose flags specify the hyperparameters of the model. You may also need to set Theano flags through the THEANO_FLAGS environment variable, which specifies compilation mode (set it to `fast_compile` during development, and delete it to use the default state for longer runs), `device`, which can be set to `cpu` or `gpu#`, and `cuda.root`, which specifies the location of CUDA when running on GPU. `floatX` should always be set to `float32`.

Here's a sample command that runs a fast, low-dimensional CPU training run, training and testing only on the dev set. It assumes that you have a copy of [SNLI](http://nlp.stanford.edu/projects/snli/) available locally.

    PYTHONPATH=spinn/python \
        THEANO_FLAGS=optimizer=fast_compile,device=cpu,floatX=float32 \
        python2.7 -m spinn.models.fat_classifier --data_type snli \
        --training_data_path snli_1.0/snli_1.0_dev.jsonl \
        --eval_data_path snli_1.0/snli_1.0_dev.jsonl \
        --embedding_data_path spinn/python/spinn/tests/test_embedding_matrix.5d.txt \
        --word_embedding_dim 5 --model_dim 10

For full runs, you'll also need a copy of the 840B word 300D [GloVe word vectors](http://nlp.stanford.edu/projects/glove/).

## C++ code

The C++ code lives in the `cpp` folder. This code implements a basic SPINN feedforward. (This implementation corresponds to the bare SPINN-PI-NT, "parsed input / no tracking" model, described in the paper.) It has been verified to produce the exact same output as a recursive neural network with the same weights and inputs. (We used a simplified version of Ozan Irsoy's [`deep-recursive` project][5] as a comparison.)

The main binary, `stacktest`, simply generates random input data and runs a feedforward. It outputs the total feedforward time elapsed and the numerical result of the feedforward.

### Dependencies

The only external dependency of the C++ code is CUDA >=7.0. The tests depend on the [`googletest` library][4], included in this repository as a Git submodule.

### Installation

First install CUDA >=7.0 and ensure that `nvcc` is on your `PATH`. Then:

    # From project root
    cd cpp

    # Pull down Git submodules (libraries)
    git submodule update --init

    # Compile C++ code
    make stacktest
    make rnntest

This should generate a binary in `cpp/bin/stacktest`.

### Running

The binary `cpp/bin/stacktest` runs on random input data. You can time the feedforward yourself by running the following commands:

    # From project root
    cd cpp

    BATCH_SIZE=512 ./bin/stacktest

You can of course set `BATCH_SIZE` to whatever integer you desire. The other model architecture parameters are fixed in the code, but you can easily change them as well [on this line][6] if you desire.

#### Baseline RNN

The binary `cpp/bin/rnntest` runs a vanilla RNN (ReLU activations) with random input data. You can run this performance test script as follows:

    # From project root
    cd cpp

    BATCH_SIZE=512 ./bin/rnntest

## License

Copyright 2016, Stanford University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

[1]: http://arxiv.org/abs/1603.06021
[2]: https://github.com/stanfordnlp/spinn/blob/master/requirements.txt
[3]: https://github.com/hans/theano-hacked/tree/8964f10e44bcd7f21ae74ea7cdc3682cc7d3258e
[4]: https://github.com/google/googletest
[5]: https://github.com/oir/deep-recursive
[6]: https://github.com/stanfordnlp/spinn/blob/5d4257f4cd15cf7213d2ff87f6f3d7f6716e2ea1/cpp/bin/stacktest.cc#L33
[7]: https://github.com/stanfordnlp/spinn/releases/tag/ACL2016
