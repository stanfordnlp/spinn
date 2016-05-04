# Stack-augmented Parser-Interpreter Neural Network

*NOTE:* This repository may be broken at head. A working snapshot is available here: https://github.com/hans/spinn/tree/deadline

This repository contains the source code described in our paper [A Fast Unified Model for Sentence Parsing and Understanding][1]. There are three separate implementations available:

- A **Python/Theano** implementation of SPINN using a naïve stack representation (named `fat-stack`)
- A **Python/Theano** implementation of SPINN using the `thin-stack` representation described in our paper
- A **C++/CUDA** implementation of the SPINN feedforward, used for performance testing

## Python code

The Python code lives, quite intuitively, in the `python` folder. We used this code to train and test the SPINN models before publication.

There is one enormous difference in the `fat-` and `thin-stack` implementations: `fat-stack` uses Theano's automatically generated symbolic backpropagation graphs, while `thin-stack` generates its own optimal backpropagation graph. This makes `thin-stack` oodles faster than its brother, but we have not yet implemented all SPINN variants to support this custom backpropagation.

### Dependencies

The Python code uses **Python 2.7** with the Theano symbolic math library.
Full requirements are listed in [`requirements.txt`][2].

We use a modified version of Theano in order to support fast forward- and backward-prop in `thin-stack` (see the [`theano-hacked` repository][3]). While it isn't absolutely necessary to use this hacked Theano, it greatly improves `thin-stack` performance.

### Running the code

TODO

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

This should generate a binary in `cpp/bin/stacktest`.

### Running

The binary `cpp/bin/stacktest` runs on random input data. You can time the feedforward yourself by running the following command:

    # From project root
    cd cpp

    BATCH_SIZE=512 ./bin/stacktest

You can of course set `BATCH_SIZE` to whatever integer you desire. The other model architecture parameters are fixed in the code, but you can easily change them as well [on this line][6] if you desire.

## Pretrained models

TODO

## License

TODO

[1]: http://arxiv.org/abs/1603.06021
[2]: https://github.com/hans/spinn/blob/master/requirements.txt
[3]: https://github.com/hans/theano-hacked/tree/8964f10e44bcd7f21ae74ea7cdc3682cc7d3258e
[4]: https://github.com/google/googletest
[5]: https://github.com/oir/deep-recursive
[6]: https://github.com/hans/spinn/blob/5d4257f4cd15cf7213d2ff87f6f3d7f6716e2ea1/cpp/bin/stacktest.cc#L33
