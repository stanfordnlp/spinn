"""From the project root directory (containing data files), this can be run with:
python rembed/models/boolean.py --training_data_path bl_train.tsv \
       --eval_data_path bl_dev.tsv
"""

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


def crop_and_pad(dataset, length):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    for example in dataset:
        padding_amount = length - len(example["op_sequence"])
        if padding_amount < 0:
            print "Cropping len " + str(len(example["op_sequence"]))
            example["op_sequence"] = example["op_sequence"][-padding_amount:-1]
        else:
            example["op_sequence"] = [0] * \
                padding_amount + example["op_sequence"]
    return dataset


def load_data(path, vocabulary=None, seq_length=None, eval_mode=False):
    dataset = import_data.import_binary_bracketed_data(path)

    if not vocabulary:
        # Build vocabulary from data
        # TODO(SB): Use a fixed vocab file in case this takes especially long, or we want
        # to include vocab items that don't appear in the training data.
        vocabulary = {import_data.REDUCE_OP: -1,
                      '*PADDING*': 0}
        types = set(itertools.chain.from_iterable([example["op_sequence"]
                                                   for example in dataset]))
        types.remove(import_data.REDUCE_OP)
        vocabulary.update({type: i + 1 for i, type in enumerate(types)})

    # Convert token sequences to integer sequences
    dataset = tokens_to_ids(vocabulary, dataset)
    dataset = crop_and_pad(dataset, FLAGS.seq_length)
    X = np.array([example["op_sequence"] for example in dataset],
                 dtype=np.int32)
    y = np.array([0 if example["label"] == "F" else 1 for example in dataset],
                 dtype=np.int32)

    logging.info("Loaded %i examples to sequences of length %i",
                 len(dataset), FLAGS.seq_length)

    # Build batched data iterator.
    if eval_mode:
        data_iter = util.MakeEvalIterator(X, y, FLAGS.batch_size)
    else:
        data_iter = util.MakeTrainingIterator(X, y, FLAGS.batch_size)

    return data_iter, vocabulary


def train():
    # Load the data
    training_data_iter, vocabulary = load_data(
        FLAGS.training_data_path, seq_length=FLAGS.seq_length)
    eval_data_iter, _ = load_data(
        FLAGS.eval_data_path, vocabulary=vocabulary, seq_length=FLAGS.seq_length, eval_mode=True)

    # Account for the *REDUCE* trigger token.
    vocab_size = len(vocabulary) - 1

    # Set up the placeholders.
    X = T.imatrix("X")
    y = T.ivector("y")
    lr = T.scalar("lr")

    logging.info("Building model.")
    vs = util.VariableStore(
        default_initializer=util.UniformInitializer(FLAGS.init_range))
    logits = build_model(vocab_size, FLAGS.seq_length, X, vs)
    xent_cost = build_cost(logits, y)

    # Set up L2 regularization.
    # TODO(SB): Don't naively L2ify the embedding matrix. It'll break on NL.
    l2_cost = 0.0
    for value in vs.vars.values():
        l2_cost += FLAGS.l2_lambda * T.sum(T.sqr(value))
    total_cost = xent_cost + l2_cost

    # Set up optimization.
    new_values = util.SGD(total_cost, vs.vars.values(), lr)
    update_fn = theano.function(
        [X, y, lr], [total_cost, xent_cost, l2_cost], updates=new_values)
    eval_fn = theano.function([X, y], xent_cost)

    # Main training loop.
    for step in range(FLAGS.training_steps):
        X_batch, y_batch = training_data_iter.next()
        total_cost_value, xent_cost_value, l2_cost_value = update_fn(
            X_batch, y_batch, FLAGS.learning_rate)
        if step % FLAGS.statistics_interval_steps == 0:
            print "Step: %i\tCost: %f %f %f" % (step, total_cost_value,
                                                xent_cost_value, l2_cost_value)
        if step % FLAGS.eval_interval_steps == 0:
            # Evaluate
            cost_accum = 0.0
            eval_batches = 0.0
            for (eval_X_batch, eval_y_batch) in eval_data_iter:
                cost_accum += eval_fn(eval_X_batch, eval_y_batch)
                eval_batches += 1.0
            print "Step: %i\tEval cost: %f" % (step, cost_accum / eval_batches)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Data settings
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "")
    gflags.DEFINE_integer("seq_length", 29, "")

    # Model architecture settings
    gflags.DEFINE_integer("embedding_dim", 5, "")
    gflags.DEFINE_integer("num_composition_layers", 1, "")

    # Optimization settings
    gflags.DEFINE_integer("training_steps", 50000, "")
    gflags.DEFINE_integer("batch_size", 32, "")
    gflags.DEFINE_float("learning_rate", 0.1, "")
    gflags.DEFINE_float("clipping_max_norm", 1.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-6, "")
    gflags.DEFINE_float("init_range", 0.1, "")

    # Display settings
    gflags.DEFINE_integer("statistics_interval_steps", 50, "")
    gflags.DEFINE_integer("eval_interval_steps", 50, "")

    # Parse command line flags
    FLAGS(sys.argv)

    train()
