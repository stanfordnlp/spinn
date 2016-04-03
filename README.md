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
Full requirements are listed in [`requirements.txt`][2].

We use a modified version of Theano in order to support fast forward- and backward-prop in `thin-stack` (see the [`theano-hacked` repository][3]). While it isn't absolutely necessary to use this hacked Theano, it greatly improves `thin-stack` performance.

### Running the code

TODO

## C++ code

TODO

## Pretrained models

TODO

## License

TODO

[1]: http://arxiv.org/abs/1603.06021
[2]: https://github.com/hans/spinn/blob/master/requirements.txt
[3]: https://github.com/hans/theano-hacked/tree/8964f10e44bcd7f21ae74ea7cdc3682cc7d3258e
