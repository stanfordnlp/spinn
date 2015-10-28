"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m rembed.models.classifier --training_data_path bl-data/bl_train.tsv \
       --eval_data_path bl-data/bl_dev.tsv

SST sentiment:
python -m rembed.models.classifier --data_type sst --embedding_dim 25 --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt
"""

from functools import partial
import pprint
import sys

import gflags
from theano import tensor as T
import theano

from rembed import afs_safe_logger
from rembed import util
from rembed.data.boolean import load_boolean_data
from rembed.data.sst import load_sst_data

from rembed.stack import HardStack, Model1


FLAGS = gflags.FLAGS


def build_model(vocab_size, seq_length, inputs, transitions, num_classes, vs,
                use_predictions=False):
    # Prepare MLP which performs stack element composition.
    compose_network = partial(
        util.MLP,
        hidden_dims=[FLAGS.embedding_dim] * (FLAGS.num_composition_layers - 1))

    # Build hard stack which scans over input sequence.
    stack = Model1(FLAGS.embedding_dim, vocab_size, seq_length,
        compose_network, vs, predict_network=util.Linear,
        use_predictions=use_predictions, X=inputs, transitions=transitions)

    # Extract top element of final stack timestep.
    embeddings = stack.final_stack[:, 0].reshape((-1, FLAGS.embedding_dim))

    # Feed forward through a single output layer
    logits = util.Linear(
        embeddings, FLAGS.embedding_dim, num_classes, vs, use_bias=True)

    return stack.transitions_pred, logits


def build_cost(logits, targets):
    # Clip gradients coming from the cost function.
    logits = theano.gradient.grad_clip(
        logits, -1 * FLAGS.clipping_max_norm, FLAGS.clipping_max_norm)

    predicted_dist = T.nnet.softmax(logits)
    costs = T.nnet.categorical_crossentropy(predicted_dist, targets)
    cost = costs.mean()

    pred = T.argmax(logits, axis=1)
    acc = 1 - T.mean(T.neq(pred, targets))

    return cost, acc


def build_action_cost(actions, targets):
    """
    Calculate cross-entropy cost over a batch of parser action sequences.
    """

    # swap seq_length dimension to front so that we can scan per timestep
    actions = T.swapaxes(actions, 0, 1)
    targets = targets.T

    def cost_t(pred, tgt):
        pred = T.nnet.softmax(pred)
        return T.nnet.categorical_crossentropy(pred, tgt)
    cost = theano.scan(cost_t, [actions, targets])[0]

    # Take mean over both example- and timestep-dimensions
    return cost.mean()


def train():
    logger = afs_safe_logger.Logger(FLAGS.experiment_name + ".log")

    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    else:
        logging.error("Bad data type.")
        return

    pp = pprint.PrettyPrinter(indent=4)
    logger.Log("Flag values:\n" + pp.pformat(FLAGS.FlagValuesDict()))

    # Load the data
    training_data_iter, vocabulary = data_manager.load_data(
        FLAGS.training_data_path, seq_length=FLAGS.seq_length, batch_size=FLAGS.batch_size, separate_transitions=True, logger=logger)

    eval_sets = []
    for eval_filename in FLAGS.eval_data_path.split(","):
        eval_data_iter, _ = data_manager.load_data(
            eval_filename, vocabulary=vocabulary, seq_length=FLAGS.seq_length, batch_size=FLAGS.batch_size,
            separate_transitions=True, eval_mode=True)
        eval_sets.append((eval_filename, eval_data_iter))

    # Account for the *REDUCE* trigger token.
    vocab_size = len(vocabulary) - 1

    # Set up the placeholders.
    X = T.imatrix("X")
    y = T.ivector("y")
    transitions = T.imatrix("transitions")
    lr = T.scalar("lr")

    logger.Log("Building model.")
    vs = util.VariableStore(
        default_initializer=util.UniformInitializer(FLAGS.init_range), logger=logger)
    actions, logits = build_model(
        vocab_size, FLAGS.seq_length, X, transitions, data_manager.NUM_CLASSES, vs,
        use_predictions=FLAGS.use_predictions)
    xent_cost, acc = build_cost(logits, y)
    action_cost = build_action_cost(actions, transitions)

    # Set up L2 regularization.
    # TODO(SB): Don't naively L2ify the embedding matrix. It'll break on NL.
    l2_cost = 0.0
    for var in vs.vars:
        if "embedding" not in var:
            l2_cost += FLAGS.l2_lambda * T.sum(T.sqr(vs.vars[var]))

    total_cost = xent_cost + l2_cost
    if FLAGS.use_predictions:
        total_cost += action_cost

    # Set up optimization.
    new_values = util.momentum(total_cost, vs.vars.values(), lr,
                               FLAGS.momentum)
    update_fn = theano.function(
        [X, y, transitions, lr],
        [total_cost, xent_cost, action_cost, l2_cost, acc],
        updates=new_values)
    eval_fn = theano.function([X, transitions, y], acc)

    # Main training loop.
    for step in range(FLAGS.training_steps):
        X_batch, y_batch, transition_batch = training_data_iter.next()
        ret = update_fn(X_batch, y_batch, transition_batch,
            FLAGS.learning_rate)
        total_cost_val, xent_cost_val, action_cost_val, l2_cost_val, acc = ret

        if step % FLAGS.statistics_interval_steps == 0:
            logger.Log(
                "Step: %i\tAcc: %f\tCost: %f %f %f %f"
                % (step, acc, total_cost_val, xent_cost_val, action_cost_val,
                   l2_cost_val))
        if step % FLAGS.eval_interval_steps == 0:
            for eval_set in eval_sets:
                # Evaluate
                acc_accum = 0.0
                eval_batches = 0.0
                for (eval_X_batch, eval_y_batch, eval_transitions_batch) in eval_set[1]:
                    acc_accum += eval_fn(eval_X_batch, eval_transitions_batch,
                                         eval_y_batch)
                    eval_batches += 1.0
                logger.Log("Step: %i\tEval acc: %f\t%s" %
                          (step, acc_accum / eval_batches, eval_set[0]))

if __name__ == '__main__':
    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "experiment", "")

    # Data types.
    gflags.DEFINE_string("data_type", "bl", "Values: bl, sst")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "")
    gflags.DEFINE_integer("seq_length", 29, "")

    # Model architecture settings.
    gflags.DEFINE_integer("embedding_dim", 5, "")
    gflags.DEFINE_integer("num_composition_layers", 1, "")
    gflags.DEFINE_bool("use_predictions", False,
                       "Use the model's predicted stack operations (rather "
                       "than ground-truth) to manipulate the stack")

    # Optimization settings.
    gflags.DEFINE_integer("training_steps", 50000, "")
    gflags.DEFINE_integer("batch_size", 32, "")
    gflags.DEFINE_float("learning_rate", 0.01, "")
    gflags.DEFINE_float("momentum", 0.9, "")
    gflags.DEFINE_float("clipping_max_norm", 1.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.1, "")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 50, "")
    gflags.DEFINE_integer("eval_interval_steps", 50, "")

    # Parse command line flags.
    FLAGS(sys.argv)

    # Run.
    train()
