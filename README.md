# Stack-augmented Parser-Interpreter Neural Network

This repository contains the source code based on the paper [A Fast Unified Model for Sentence Parsing and Understanding][1] and [original codebase][9]. For a more informal introduction to the ideas behind the model, see this [Stanford NLP blog post][8].


The included implementations are:

- A **Python/Pytorch** implementation of SPINN using a naïve stack representation (named `fat-stack`)

## Python code

The Python code lives, quite intuitively, in the `python` folder. We used this code to train and test the SPINN models before publication.

### Installation

Requirements:

- Python 2.7
- Pytorch

Install most required Python dependencies using the command below.

    pip install -r python/requirements.txt

Install Pytorch based on instructions online: http://pytorch.org

### Running the code

The main executable for the SNLI experiments in the paper is [supervised_classifier.py](https://github.com/mrdrozdov/spinn/blob/master/python/spinn/models/supervised_classifier.py), whose flags specify the hyperparameters of the model. You can specify gpu usage by setting `--gpu` flag greater than or equal to 0. Uses the CPU by default.

Here's a sample command that runs a fast, low-dimensional CPU training run, training and testing only on the dev set. It assumes that you have a copy of [SNLI](http://nlp.stanford.edu/projects/snli/) available locally.

    PYTHONPATH=spinn/python \
        python2.7 -m spinn.models.supervised_classifier --data_type snli \
        --training_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --eval_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --embedding_data_path python/spinn/tests/test_embedding_matrix.5d.txt \
        --word_embedding_dim 5 --model_dim 10 --model_type CBOW

For full runs, you'll also need a copy of the 840B word 300D [GloVe word vectors](http://nlp.stanford.edu/projects/glove/).

## Log Analysis

This project contains a handful of tools for easier analysis of your model's performance.

For one, after a periodic number of batches, some useful statistics are printed to a file specified by `--log_path`. This is convenient for visual inspection, and the script [parse_logs.py](https://github.com/mrdrozdov/spinn/blob/master/scripts/parse_logs.py) is an example of how to easily parse this log file.

In addition, there is support for realtime summaries using [Visdom](https://github.com/facebookresearch/visdom). This requires a few steps:

1. Run your experiment normally, but specify a `--metrics_path`.
2. Run Visdom in it's own terminal instance: `python -m visdom.server`
3. Run this project's [visdom_reporter.py](https://github.com/mrdrozdov/spinn/blob/master/scripts/visdom_reporter.py) script, specifying a root which matches the `--metrics_path` flag: `python scripts/visdom_reporter.py --root $METRICS_PATH`

Then open Visdom in a browser window to see graphs representing accuracy, loss and some other metrics updated in real time. This is most useful when running multiple experiments simultaneously.

## Contributing

If you're interested in proposing a change or fix to SPINN, please submit a Pull Request. In addition, ensure that existing tests pass, and add new tests as you see appropriate. To run tests, simply run this command from the root directoy:

    nosetests python/spinn/tests

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
[8]: http://nlp.stanford.edu/blog/hybrid-tree-sequence-neural-networks-with-spinn/
[9]: https://github.com/stanfordnlp/spinn
