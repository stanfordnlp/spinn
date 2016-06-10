# Stack-augmented Parser-Interpreter Neural Network

This repository contains the source code described in our paper [A Fast Unified Model for Sentence Parsing and Understanding][1]. There are three separate implementations available:

- A **Python/Theano** implementation of SPINN using a naïve stack representation (named `fat-stack`)
- A **Python/Theano** implementation of SPINN using the `thin-stack` representation described in our paper
- A **C++/CUDA** implementation of the SPINN feedforward, used for performance testing

## Python code

The Python code lives, quite intuitively, in the `python` folder. We used this code to train and test the SPINN models before publication.

There is one enormous difference in the `fat-` and `thin-stack` implementations: `fat-stack` uses Theano's automatically generated symbolic backpropagation graphs, while `thin-stack` generates its own optimal backpropagation graph. This makes `thin-stack` oodles faster than its brother, but we have not yet implemented all SPINN variants to support this custom backpropagation.

### Dependencies

The Python code uses **Python 2.7** with the Theano symbolic math library.
Full Python package requirements are listed in [`requirements.txt`][2].

We use a modified version of Theano in order to support fast forward- and backward-prop in `thin-stack` (see the [`theano-hacked` repository][3]). While it isn't absolutely necessary to use this hacked Theano, it greatly improves `thin-stack` performance.

To install everything you need, run the following to get our requirements specification (also available from GitHub) and install everything in it. *WARNING:* This installs our 
custom branch of Theano, so it may be dangerous to run this in an environment where you'll also be running code that depends upon stock Theano.

    wget https://raw.githubusercontent.com/stanfordnlp/spinn/master/python/requirements.txt
    python -m pip install -r requirements.txt

We also require CUDA >= 7.0 with CuDNN v4. (CuDNN v5 is not compatible with our fork of Theano.)

### Running the code

The main executable for the SNLI experiments in the paper is [fat_classifier.py](https://github.com/stanfordnlp/spinn/blob/master/python/spinn/models/fat_classifier.py), whose flags specify the hyperparameters of the model. You may also need to set Theano flags through the THEANO_FLAGS environment variable, which specifies compilation mode (set it to `fast_compile` during development, and delete it to use the default state for longer runs), `device`, which can be set to `cpu` or `gpu#`, and `cuda.root`, which specifies the location of CUDA when running on GPU. `floatX` should always be set to `float32`.

Here's a sample command that runs a fast, low-dimensional CPU training run, training and testing only on the dev set. It assumes that you have a copy of [SNLI](http://nlp.stanford.edu/projects/snli/) available locally.

    PYTHONPATH=spinn/python THEANO_FLAGS=optimizer=fast_compile,device=cpu,compiledir=/tmp/theano.tmp,floatX=float32 python2.7 -m spinn.models.fat_classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path spinn/python/spinn/tests/test_embedding_matrix.5d.txt --word_embedding_dim 5 --model_dim 10

For full runs, you'll also need a copy of the 840B word 300D [GloVe word vectors](http://nlp.stanford.edu/projects/glove/).

The commands to reproduce our main results can be found [here](https://github.com/stanfordnlp/spinn/blob/master/checkpoints/commands.sh).



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

## Pretrained models

TODO

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
