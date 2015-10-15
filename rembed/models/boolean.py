import itertools
import logging
import sys

import gflags
import numpy as np
from theano import tensor as T
import theano

from rembed import util
from rembed.data.boolean import import_binary_bracketed_data as import_data
from rembed.stack import HardStack


FLAGS = gflags.FLAGS


def build_model(vocab_size, seq_length, inputs, vs):
    # Build hard stack which scans over input sequence.
    stack = HardStack(
        FLAGS.embedding_dim, vocab_size, seq_length, vs, X=inputs)

    # Extract top element of final stack timestep.
    embeddings = stack.final_stack[:, 0].reshape((-1, FLAGS.embedding_dim))

    # Feed forward through a single output layer
    outputs = util.Linear(
        embeddings, FLAGS.embedding_dim, 1, vs, use_bias=True)
    outputs = T.nnet.sigmoid(outputs)
    outputs = outputs.reshape((-1,))
    return outputs


def build_cost(outputs, targets):
    costs = T.nnet.binary_crossentropy(outputs, targets)
    cost = costs.mean()
    return cost


def tokens_to_ids(vocabulary, dataset):
    """Replace strings in original boolean dataset with token IDs."""

    for example in dataset:
        example["op_sequence"] = [vocabulary[token]
                                  for token in example["op_sequence"]]
    return dataset


def load_data():
    dataset = import_data.import_binary_bracketed_data(FLAGS.data_path)

    # Force a particular seq length
    seq_length = FLAGS.seq_length
    dataset = [example for example in dataset
               if len(example["op_sequence"]) == seq_length]
    logging.info("Retained %i examples of length %i", len(dataset), seq_length)

    # Build vocabulary from data
    vocabulary = {import_data.REDUCE_OP: -1}
    types = set(itertools.chain.from_iterable([example["op_sequence"]
                                               for example in dataset]))
    types.remove(import_data.REDUCE_OP)
    vocabulary.update({type: i for i, type in enumerate(types)})

    # Convert token sequences to integer sequences
    dataset = tokens_to_ids(vocabulary, dataset)
    X = np.array([example["op_sequence"] for example in dataset],
                 dtype=np.int32)
    y = np.array([0 if example["label"] == "F" else 1 for example in dataset],
                 dtype=np.int32)

    # Build batched data iterator.
    def data_iter():
        size = FLAGS.batch_size
        start = -1 * size

        # TODO Don't be lazy and drop remainder of examples that don't fit into
        # a final batch
        while True:
            start += size
            if start > len(X):
                start = 0
            yield X[start:start + size], y[start:start + size]

    return data_iter(), len(vocabulary) - 1, seq_length


def train():
    data_iter, vocab_size, seq_length = load_data()

    X = T.imatrix("X")
    y = T.ivector("y")
    lr = T.scalar("lr")

    logging.info("Building model.")
    vs = util.VariableStore()
    model_outputs = build_model(vocab_size, seq_length, X, vs)
    cost = build_cost(model_outputs, y)

    updates = util.SGD(cost, vs.vars.values(), lr)
    update_fn = theano.function([X, y, lr], cost, updates=updates)

    for step in range(FLAGS.training_steps):
        X_batch, y_batch = data_iter.next()

        print update_fn(X_batch, y_batch, FLAGS.learning_rate)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Data settings
    gflags.DEFINE_string("data_path", None, "")
    gflags.DEFINE_integer("seq_length", 11, "")

    # Model architecture settings
    gflags.DEFINE_integer("embedding_dim", 20, "")

    # Optimization settings
    gflags.DEFINE_integer("training_steps", 500, "")
    gflags.DEFINE_integer("batch_size", 32, "")
    gflags.DEFINE_float("learning_rate", 0.1, "")

    # Parse command line flags
    FLAGS(sys.argv)

    train()
