"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m spinn.models.fat_classifier --training_data_path ../bl-data/pbl_train.tsv \
       --eval_data_path ../bl-data/pbl_dev.tsv

SST sentiment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type sst --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

SNLI entailment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl \
       --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

Note: If you get an error starting with "TypeError: ('Wrong number of dimensions..." during development,
    there may already be a saved checkpoint in ckpt_path that matches the name of the model you're developing.
    Move or delete it as appropriate.
"""

import os
import json
import math
import random
import pprint
import sys
import time
from collections import deque

import gflags
import numpy as np

from spinn import util
from spinn.util import afs_safe_logger
from spinn.data.arithmetic import load_sign_data
from spinn.data.arithmetic import load_simple_data
from spinn.data.dual_arithmetic import load_eq_data
from spinn.data.dual_arithmetic import load_relational_data
from spinn.data.boolean import load_boolean_data
from spinn.data.listops import load_listops_data
from spinn.data.sst import load_sst_data, load_sst_binary_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import ModelTrainer, the_gpu, to_gpu, l2_cost, flatten, debug_gradient
from spinn.util.misc import Accumulator, MetricsLogger, EvalReporter, time_per_token
from spinn.util.misc import recursively_set_device
from spinn.util.logging import train_format, train_extra_format, train_stats, train_accumulate
from spinn.util.loss import auxiliary_loss
import spinn.util.evalb as evalb

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

from spinn.models.base import get_data_manager, init_model
from spinn.models.base import flag_defaults, get_flags
from spinn.models.base import sequential_only, get_batch, truncate


FLAGS = gflags.FLAGS


def evaluate(model, eval_set, logger, step, vocabulary=None):
    filename, dataset = eval_set

    reporter = EvalReporter()

    # Evaluate
    class_correct = 0
    class_total = 0
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    start = time.time()

    model.eval()

    transition_preds = []
    transition_targets = []

    for i, batch in enumerate(dataset):
        eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch = get_batch(batch)[:4]

        # Run model.
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions)

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()
        pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
        class_correct += pred.eq(target).sum()
        class_total += target.size(0)

        # Optionally calculate transition loss/acc.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        total_tokens += sum([(nt+1)/2 for nt in eval_num_transitions_batch.reshape(-1)])

        # Accumulate stats for transition accuracy.
        if transition_loss is not None:
            transition_preds.append([m["t_preds"] for m in model.spinn.memories])
            transition_targets.append([m["t_given"] for m in model.spinn.memories])

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    end = time.time()
    total_time = end - start

    # Get time per token.
    time_metric = time_per_token([total_tokens], [total_time])

    # Get class accuracy.
    eval_class_acc = class_correct / float(class_total)

    # Get transition accuracy if applicable.
    if len(transition_preds) > 0:
        all_preds = np.array(flatten(transition_preds))
        all_truth = np.array(flatten(transition_targets))
        eval_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])
    else:
        eval_trans_acc = 0.0

    stats_str = "Step: %i Eval acc: %f %f %s Time: %5f" % (step, eval_class_acc, eval_trans_acc, filename, time_metric)

    # Extra Component.
    stats_str += "\nEval Extra:"

    logger.Log(stats_str)

    return eval_class_acc


def get_checkpoint_path(ckpt_path, experiment_name, suffix=".ckpt", best=False):
    # Set checkpoint path.
    if ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".ckpt_best"):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = os.path.join(ckpt_path, experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
    return checkpoint_path


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    # Select data format.
    data_manager = get_data_manager(FLAGS.data_type)

    logger.Log("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path, FLAGS.lowercase)

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            raw_eval_data, _ = data_manager.load_data(eval_filename, FLAGS.lowercase)
            raw_eval_sets.append((eval_filename, raw_eval_data))

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        train_embeddings = False
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
        train_embeddings = True

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

    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(data_manager.LABEL_MAP)

    model, optimizer, trainer = init_model(FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager)

    # Build trainer.
    trainer = ModelTrainer(model, optimizer)

    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name, best=True)

    # Load checkpoint if available.
    if FLAGS.load_best and os.path.isfile(best_checkpoint_path):
        logger.Log("Found best checkpoint, restoring.")
        step, best_dev_error = trainer.load(best_checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    elif os.path.isfile(standard_checkpoint_path):
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = trainer.load(standard_checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    # GPU support.
    the_gpu.gpu = FLAGS.gpu
    if FLAGS.gpu >= 0:
        model.cuda()
    else:
        model.cpu()
    recursively_set_device(optimizer.state_dict(), the_gpu.gpu)

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)

    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)

    # Do an evaluation-only run.
    if only_forward:
        for index, eval_set in enumerate(eval_iterators):
            acc = evaluate(model, eval_set, logger, step, vocabulary)
    else:
        # Build log format strings.
        model.train()
        X_batch, transitions_batch, y_batch, num_transitions_batch = get_batch(training_data_iter.next())[:4]
        model(X_batch, transitions_batch, y_batch,
                use_internal_parser=FLAGS.use_internal_parser,
                validate_transitions=FLAGS.validate_transitions
                )

        train_str = train_format(model)
        logger.Log("Train-Format: {}".format(train_str))
        train_extra_str = train_extra_format(model)
        logger.Log("Train-Extra-Format: {}".format(train_extra_str))

         # Train
        logger.Log("Training.")

        # New Training Loop
        progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
        progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

        for step in range(step, FLAGS.training_steps):
            model.train()

            start = time.time()

            batch = get_batch(training_data_iter.next())
            X_batch, transitions_batch, y_batch, num_transitions_batch = batch[:4]

            total_tokens = sum([(nt+1)/2 for nt in num_transitions_batch.reshape(-1)])

            # Reset cached gradients.
            optimizer.zero_grad()

            if FLAGS.model_type == "RLSPINN":
                model.spinn.epsilon = FLAGS.rl_epsilon * math.exp(-step/FLAGS.rl_epsilon_decay)

            # Run model.
            output = model(X_batch, transitions_batch, y_batch,
                use_internal_parser=FLAGS.use_internal_parser,
                validate_transitions=FLAGS.validate_transitions
                )

            # Normalize output.
            logits = F.log_softmax(output)

            # Calculate class accuracy.
            target = torch.from_numpy(y_batch).long()
            pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability
            class_acc = pred.eq(target).sum() / float(target.size(0))

            # Calculate class loss.
            xent_loss = nn.NLLLoss()(logits, to_gpu(Variable(target, volatile=False)))

            # Optionally calculate transition loss.
            transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

            # Extract L2 Cost
            l2_loss = l2_cost(model, FLAGS.l2_lambda) if FLAGS.use_l2_cost else None

            # Accumulate Total Loss Variable
            total_loss = 0.0
            total_loss += xent_loss
            if l2_loss is not None:
                total_loss += l2_loss
            if transition_loss is not None and model.optimize_transition_loss:
                total_loss += transition_loss
            total_loss += auxiliary_loss(model)

            # Backward pass.
            total_loss.backward()

            # Hard Gradient Clipping
            clip = FLAGS.clipping_max_value
            for p in model.parameters():
                if p.requires_grad:
                    p.grad.data.clamp_(min=-clip, max=clip)

            # Learning Rate Decay
            if FLAGS.actively_decay_learning_rate:
                optimizer.lr = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

            # Gradient descent step.
            optimizer.step()

            end = time.time()

            total_time = end - start

            train_accumulate(model, data_manager, A, batch)
            A.add('class_acc', class_acc)
            A.add('total_tokens', total_tokens)
            A.add('total_time', total_time)

            if step % FLAGS.statistics_interval_steps == 0:
                progress_bar.step(i=FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)
                progress_bar.finish()

                A.add('xent_cost', xent_loss.data[0])
                A.add('l2_cost', l2_loss.data[0])
                stats_args = train_stats(model, optimizer, A, step)

                logger.Log(train_str.format(**stats_args))
                logger.Log(train_extra_str.format(**stats_args))

            if step > 0 and step % FLAGS.eval_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(model, eval_set, logger, step)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > FLAGS.ckpt_step:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        trainer.save(best_checkpoint_path, step, best_dev_error)
                progress_bar.reset()

            if step > FLAGS.ckpt_step and step % FLAGS.ckpt_interval_steps == 0:
                logger.Log("Checkpointing.")
                trainer.save(standard_checkpoint_path, step, best_dev_error)

            progress_bar.step(i=step % FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS)

    run(only_forward=FLAGS.expanded_eval_only_mode)
