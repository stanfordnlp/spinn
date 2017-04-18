import sys
import os
import json
import math
import random
import time

import gflags
import numpy as np

from spinn import util
from spinn.data.arithmetic import load_sign_data
from spinn.data.arithmetic import load_simple_data
from spinn.data.dual_arithmetic import load_eq_data
from spinn.data.dual_arithmetic import load_relational_data
from spinn.data.boolean import load_boolean_data
from spinn.data.listops import load_listops_data
from spinn.data.sst import load_sst_data, load_sst_binary_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import ModelTrainer, the_gpu, to_gpu, l2_cost, flatten, bundle
from spinn.util.blocks import Linear, LSTM, EncodeGRU, ReduceTreeLSTM, ReduceTreeGRU, IntraAttention
from spinn.util.misc import Accumulator, EvalReporter, time_per_token
from spinn.util.misc import recursively_set_device, Args
from spinn.util.metrics import MetricsWriter
from spinn.util.logging import train_format, train_extra_format, train_stats, train_accumulate
from spinn.util.logging import eval_format, eval_extra_format
from spinn.util.loss import auxiliary_loss
import spinn.util.evalb as evalb
from spinn.util.logparse import parse_flags

import spinn.rl_spinn
import spinn.fat_stack
import spinn.plain_rnn
import spinn.cbow

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


FLAGS = gflags.FLAGS


def sequential_only():
    return FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW"


def log_path(FLAGS, load=False):
    lp = FLAGS.load_log_path if load else FLAGS.log_path
    en = FLAGS.load_experiment_name if load else FLAGS.experiment_name
    return os.path.join(lp, en) + ".log"


def get_batch(batch):
    X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids = batch

    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = sequential_only()
    transitions_batch_is_left_padded = True
    max_length = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    # Truncate batch.
    X_batch = truncate(X_batch, seq_length, max_length, X_batch_is_left_padded)
    transitions_batch = truncate(transitions_batch, seq_length, max_length, transitions_batch_is_left_padded)

    return X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids


def truncate(data, seq_length, max_length, left_padded):
    if left_padded:
        data = data[:, seq_length - max_length:]
    else:
        data = data[:, :max_length]
    return data


def get_data_manager(data_type):
    # Select data format.
    if data_type == "bl":
        data_manager = load_boolean_data
    elif data_type == "sst":
        data_manager = load_sst_data
    elif data_type == "sst-binary":
        data_manager = load_sst_binary_data
    elif data_type == "snli":
        data_manager = load_snli_data
    elif data_type == "arithmetic":
        data_manager = load_simple_data
    elif data_type == "listops":
        data_manager = load_listops_data
    elif data_type == "sign":
        data_manager = load_sign_data
    elif data_type == "eq":
        data_manager = load_eq_data
    elif data_type == "relational":
        data_manager = load_relational_data
    else:
        raise NotImplementedError

    return data_manager


def get_checkpoint_path(ckpt_path, experiment_name, suffix=".ckpt", best=False):
    # Set checkpoint path.
    if ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".ckpt_best"):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = os.path.join(ckpt_path, experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
    return checkpoint_path


def load_data_and_embeddings(FLAGS, data_manager, logger, training_data_path, eval_data_path):

    choose_train = lambda x: True
    if FLAGS.train_genre is not None:
        choose_train = lambda x: x.get('genre') == FLAGS.train_genre

    choose_eval = lambda x: True
    if FLAGS.eval_genre is not None:
        choose_eval = lambda x: x.get('genre') == FLAGS.eval_genre

    if FLAGS.data_type == "snli":
        # Load the data.
        raw_training_data, vocabulary = data_manager.load_data(
            training_data_path, FLAGS.lowercase, choose_train)
    else:
        # Load the data.
        raw_training_data, vocabulary = data_manager.load_data(
            training_data_path, FLAGS.lowercase)


    if FLAGS.data_type == "snli":
        # Load the eval data.
        raw_eval_sets = []
        raw_eval_data, _ = data_manager.load_data(eval_data_path, FLAGS.lowercase, choose_eval)
        raw_eval_sets.append((eval_data_path, raw_eval_data))
    else:
        # Load the eval data.
        raw_eval_sets = []
        raw_eval_data, _ = data_manager.load_data(eval_data_path, FLAGS.lowercase)
        raw_eval_sets.append((eval_data_path, raw_eval_data))


    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=sequential_only())
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)

    # Preprocess eval sets.
    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        eval_data = util.PreprocessDataset(
            raw_eval_set, vocabulary,
            FLAGS.eval_seq_length if FLAGS.eval_seq_length is not None else FLAGS.seq_length,
            data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=sequential_only())
        eval_it = util.MakeEvalIterator(eval_data,
            FLAGS.batch_size, FLAGS.eval_data_limit, bucket_eval=FLAGS.bucket_eval,
            shuffle=FLAGS.shuffle_eval, rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))

    return vocabulary, initial_embeddings, training_data_iter, eval_iterators


def get_flags():
    # Debug settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_integer("deque_length", None, "Max trailing examples to use for statistics.")
    gflags.DEFINE_string("sha", "", "")
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("load_experiment_name", None, "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "sst-binary", "snli", "multinli", "arithmetic", "listops", "sign", "eq", "relational"],
        "Which data handler and classifier to use.")

    # Choose Genre.
    # 'fiction', 'government', 'slate', 'telephone', 'travel'
    # 'facetoface', 'letters', 'nineeleven', 'oup', 'verbatim'
    gflags.DEFINE_string("train_genre", None, "Filter MultiNLI data by genre.")
    gflags.DEFINE_string("eval_genre", None, "Filter MultiNLI data by genre.")

    # Where to store checkpoints
    gflags.DEFINE_string("log_path", "./logs", "A directory in which to write logs.")
    gflags.DEFINE_string("load_log_path", None, "A directory in which to write logs.")
    gflags.DEFINE_string("ckpt_path", None, "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("metrics_path", None, "A directory in which to write metrics.")
    gflags.DEFINE_integer("ckpt_step", 1000, "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_boolean("load_best", False, "If True, attempt to load 'best' checkpoint.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 200, "")
    gflags.DEFINE_integer("eval_seq_length", None, "")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_boolean("bucket_eval", True, "Bucket evaluation data for speed improvement.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer("shuffle_eval_seed", 123, "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN", ["CBOW", "RNN", "SPINN", "RLSPINN"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_boolean("lowercase", False, "When True, ignore case.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean("validate_transitions", True,
        "Constrain predicted transitions to ones that give a valid parse tree.")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
        "Used for dropout on transformed embeddings and in the encoder RNN.")
    gflags.DEFINE_boolean("use_l2_cost", True, "")
    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")

    # Tracker settings.
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", None, "Set to none to avoid using tracker.")
    gflags.DEFINE_float("transition_weight", None, "Set to none to avoid predicting transitions.")
    gflags.DEFINE_boolean("lateral_tracking", True,
        "Use previous tracker state as input for new state.")
    gflags.DEFINE_boolean("use_tracking_in_composition", True,
        "Use tracking lstm output as input for the reduce function.")
    gflags.DEFINE_boolean("predict_use_cell", True,
        "Use cell output as feature for transition net.")

    # Reduce settings.
    gflags.DEFINE_enum("reduce", "treelstm", ["treelstm", "treegru", "tanh"], "Specify composition function.")

    # Encode settings.
    gflags.DEFINE_enum("encode", "projection", ["pass", "projection", "gru", "attn"], "Encode embeddings with sequential context.")
    gflags.DEFINE_boolean("encode_reverse", False, "Encode in reverse order.")
    gflags.DEFINE_boolean("encode_bidirectional", False, "Encode in both directions.")
    gflags.DEFINE_integer("encode_num_layers", 1, "RNN layers in encoding net.")

    # RL settings.
    gflags.DEFINE_float("rl_mu", 0.1, "Use in exponential moving average baseline.")
    gflags.DEFINE_enum("rl_baseline", "ema", ["ema", "pass", "greedy", "value"],
        "Different configurations to approximate reward function.")
    gflags.DEFINE_enum("rl_reward", "standard", ["standard", "xent"],
        "Different reward functions to use.")
    gflags.DEFINE_float("rl_weight", 1.0, "Hyperparam for REINFORCE loss.")
    gflags.DEFINE_boolean("rl_whiten", False, "Reduce variance in advantage.")
    gflags.DEFINE_boolean("rl_valid", True, "Only consider non-validated actions.")
    gflags.DEFINE_boolean("rl_entropy", False, "Entropy regularization on transition policy.")
    gflags.DEFINE_float("rl_entropy_beta", 0.001, "Entropy regularization on transition policy.")
    gflags.DEFINE_float("rl_epsilon", 1.0, "Percent of sampled actions during train time.")
    gflags.DEFINE_float("rl_epsilon_decay", 50000, "Percent of sampled actions during train time.")
    gflags.DEFINE_float("rl_confidence_interval", 1000, "Penalize probabilities of transitions.")
    gflags.DEFINE_float("rl_confidence_penalty", None, "Penalize probabilities of transitions.")
    gflags.DEFINE_boolean("rl_catalan", True, "Sample over a uniform distribution of binary trees.")
    gflags.DEFINE_boolean("rl_wake_sleep", False, "Inverse relationship between temperature and rl_weight.")

    # MLP settings.
    gflags.DEFINE_integer("mlp_dim", 1024, "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "Number of MLP layers.")
    gflags.DEFINE_boolean("mlp_bn", True, "When True, batch normalization is used between MLP layers.")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
        "Used for dropout in the semantic task classifier.")

    # Optimization settings.
    gflags.DEFINE_enum("optimizer_type", "Adam", ["Adam", "RMSprop"], "")
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in optimizer.")
    gflags.DEFINE_boolean("actively_decay_learning_rate", True, "Used in optimizer.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("metrics_interval_steps", 10, "Evaluate at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")
    gflags.DEFINE_integer("sample_interval_steps", None, "Sample transitions at this interval.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")
    gflags.DEFINE_boolean("evalb", False, "Print transition statistics.")
    gflags.DEFINE_integer("num_samples", 0, "Print sampled transitions.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_boolean("write_eval_report", False, "")
    gflags.DEFINE_boolean("eval_report_use_preds", True, "If False, use the given transitions in the report, "
        "otherwise use predicted transitions. Note that when predicting transitions but not using them, the "
        "reported predictions will look very odd / not valid.")


def flag_defaults(FLAGS, load_log_flags=False):
    if load_log_flags:
        if FLAGS.load_log_path and os.path.exists(log_path(FLAGS, load=True)):
            log_flags = parse_flags(log_path(FLAGS, load=True))
            for k in log_flags.keys():
                setattr(FLAGS, k, log_flags[k])

            # Optionally override flags from log file.
            FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    if not FLAGS.load_log_path:
        FLAGS.load_log_path = FLAGS.log_path

    if not FLAGS.load_experiment_name:
        FLAGS.load_experiment_name = FLAGS.experiment_name

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.load_log_path

    if not FLAGS.sample_interval_steps:
        FLAGS.sample_interval_steps = FLAGS.statistics_interval_steps

    if not FLAGS.metrics_path:
        FLAGS.metrics_path = FLAGS.log_path

    if FLAGS.model_type == "CBOW" or FLAGS.model_type == "RNN":
        FLAGS.num_samples = 0

    if not torch.cuda.is_available():
        FLAGS.gpu = -1


def init_model(FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager):
    # Choose model.
    logger.Log("Building model.")
    if FLAGS.model_type == "CBOW":
        build_model = spinn.cbow.build_model
    elif FLAGS.model_type == "RNN":
        build_model = spinn.plain_rnn.build_model
    elif FLAGS.model_type == "SPINN":
        build_model = spinn.fat_stack.build_model
    elif FLAGS.model_type == "RLSPINN":
        build_model = spinn.rl_spinn.build_model
    else:
        raise NotImplementedError

    # Input Encoder.
    context_args = Args()
    context_args.reshape_input = lambda x, batch_size, seq_length: x
    context_args.reshape_context = lambda x, batch_size, seq_length: x
    context_args.input_dim = FLAGS.word_embedding_dim

    if FLAGS.encode == "projection":
        encoder = Linear()(FLAGS.word_embedding_dim, FLAGS.model_dim)
    elif FLAGS.encode == "gru":
        context_args.reshape_input = lambda x, batch_size, seq_length: x.view(batch_size, seq_length, -1)
        context_args.reshape_context = lambda x, batch_size, seq_length: x.view(batch_size * seq_length, -1)
        context_args.input_dim = FLAGS.model_dim
        encoder = EncodeGRU(FLAGS.word_embedding_dim, FLAGS.model_dim,
                num_layers=FLAGS.encode_num_layers,
                bidirectional=FLAGS.encode_bidirectional,
                reverse=FLAGS.encode_reverse)
    elif FLAGS.encode == "attn":
        context_args.reshape_input = lambda x, batch_size, seq_length: x.view(batch_size, seq_length, -1)
        context_args.reshape_context = lambda x, batch_size, seq_length: x.view(batch_size * seq_length, -1)
        context_args.input_dim = FLAGS.model_dim
        encoder = IntraAttention(FLAGS.word_embedding_dim, FLAGS.model_dim)
    elif FLAGS.encode == "pass":
        encoder = lambda x: x
    else:
        raise NotImplementedError

    context_args.encoder = encoder

    # Composition Function.
    composition_args = Args()
    composition_args.lateral_tracking = FLAGS.lateral_tracking
    composition_args.use_tracking_in_composition = FLAGS.use_tracking_in_composition
    composition_args.size = FLAGS.model_dim
    composition_args.tracker_size = FLAGS.tracking_lstm_hidden_dim
    composition_args.transition_weight = FLAGS.transition_weight
    composition_args.wrap_items = lambda x: torch.cat(x, 0)
    composition_args.extract_h = lambda x: x
    composition_args.extract_c = None

    if FLAGS.reduce == "treelstm":
        composition_args.wrap_items = lambda x: bundle(x)
        composition_args.extract_h = lambda x: x.h
        composition_args.extract_c = lambda x: x.c
        composition_args.size = FLAGS.model_dim / 2
        composition = ReduceTreeLSTM(FLAGS.model_dim / 2,
            FLAGS.tracking_lstm_hidden_dim,
            FLAGS.use_tracking_in_composition)
    elif FLAGS.reduce == "tanh":
        class ReduceTanh(nn.Module):
            def forward(self, lefts, rights, tracking=None):
                batch_size = len(lefts)
                ret = torch.cat(lefts, 0) + F.tanh(torch.cat(rights, 0))
                return torch.chunk(ret, batch_size, 0)
        composition = ReduceTanh()
    elif FLAGS.reduce == "treegru":
        composition = ReduceTreeGRU(FLAGS.model_dim,
            FLAGS.tracking_lstm_hidden_dim,
            FLAGS.use_tracking_in_composition)
    else:
        raise NotImplementedError

    composition_args.composition = composition

    model = build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS, context_args, composition_args)

    # Build optimizer.
    if FLAGS.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif FLAGS.optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate, eps=1e-08)
    else:
        raise NotImplementedError

    # Build trainer.
    trainer = ModelTrainer(model, optimizer)

    # Print model size.
    logger.Log("Architecture: {}".format(model))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0) for w in model.parameters()])
    logger.Log("Total params: {}".format(total_params))

    return model, optimizer, trainer
