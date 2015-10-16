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
        FLAGS.embedding_dim, vocab_size, seq_length, FLAGS.num_composition_layers, vs, X=inputs)

    # Extract top element of final stack timestep.
    embeddings = stack.final_stack[:, 0].reshape((-1, FLAGS.embedding_dim))

    # Feed forward through a single output layer
    logits = util.Linear(
        embeddings, FLAGS.embedding_dim, 2, vs, use_bias=True)

    return logits


def build_cost(logits, targets):
    # Clip gradients coming from the cost function.
    logits = theano.gradient.grad_clip(
        logits, -1 * FLAGS.clipping_max_norm, FLAGS.clipping_max_norm)

    predicted_dist = T.nnet.softmax(logits)
    costs = T.nnet.categorical_crossentropy(predicted_dist, targets)
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
    # TODO(SB): Add shuffling.
    def data_iter():
        size = FLAGS.batch_size
        start = -1 * size

        # TODO Don't be lazy and drop remainder of examples that don't fit into
        # a final batch [No need to do that if we shuffle. -SB]
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
    vs = util.VariableStore(
        default_initializer=util.UniformInitializer(FLAGS.init_range))
    logits = build_model(vocab_size, seq_length, X, vs)
    xent_cost = build_cost(logits, y)

    # L2 regularization
    # TODO(SB): Don't naively L2ify the embedding matrix. It'll break on NL.
    l2_cost = 0.0
    for value in vs.vars.values():
        l2_cost += FLAGS.l2_lambda * T.sum(T.sqr(value))
    total_cost = xent_cost + l2_cost

    new_values = util.SGD(total_cost, vs.vars.values(), lr)
    update_fn = theano.function(
        [X, y, lr], [total_cost, xent_cost, l2_cost], updates=new_values)

    for step in range(FLAGS.training_steps):
        X_batch, y_batch = data_iter.next()
        total_cost_value, xent_cost_value, l2_cost_value = update_fn(
            X_batch, y_batch, FLAGS.learning_rate)
        if step % FLAGS.statistics_interval_steps == 0:
            print "Step: %i\tCost: %f %f %f" % (step, total_cost_value, xent_cost_value, l2_cost_value)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Data settings
    gflags.DEFINE_string("data_path", None, "")
    gflags.DEFINE_integer("seq_length", 11, "")

    # Model architecture settings
    gflags.DEFINE_integer("embedding_dim", 5, "")
    gflags.DEFINE_integer("num_composition_layers", 2, "")

    # Optimization settings
    gflags.DEFINE_integer("training_steps", 50000, "")
    gflags.DEFINE_integer("batch_size", 32, "")
    gflags.DEFINE_float("learning_rate", 0.1, "")
    gflags.DEFINE_float("clipping_max_norm", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 0.0001, "")
    gflags.DEFINE_float("init_range", 0.2, "")

    # Display settings
    gflags.DEFINE_integer("statistics_interval_steps", 50, "")

    # Parse command line flags
    FLAGS(sys.argv)

    train()
