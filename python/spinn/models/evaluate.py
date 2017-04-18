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
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import the_gpu, to_gpu, l2_cost, flatten, debug_gradient
from spinn.util.misc import Accumulator, MetricsLogger, EvalReporter, time_per_token
from spinn.util.misc import recursively_set_device
from spinn.util.metrics import MetricsWriter
from spinn.util.logging import train_format, train_extra_format, train_stats, train_accumulate
from spinn.util.logging import train_metrics, eval_metrics
from spinn.util.logging import eval_format, eval_extra_format, eval_stats, eval_accumulate
from spinn.util.loss import auxiliary_loss
from spinn.util.sparks import sparks
import spinn.util.evalb as evalb

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


from spinn.models.base import get_data_manager, get_flags, get_batch
from spinn.models.base import flag_defaults, init_model
from spinn.models.base import sequential_only, get_checkpoint_path, log_path
from spinn.models.base import load_data_and_embeddings


FLAGS = gflags.FLAGS


def evaluate(FLAGS, model, data_manager, eval_set, index, logger, step, vocabulary=None):
    filename, dataset = eval_set

    A = Accumulator()
    M = MetricsWriter(os.path.join(FLAGS.metrics_path, FLAGS.experiment_name))
    reporter = EvalReporter()

    eval_str = eval_format(model)
    eval_extra_str = eval_extra_format(model)

    # Evaluate
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    invalid = 0
    start = time.time()

    model.eval()
    for i, dataset_batch in enumerate(dataset):
        batch = get_batch(dataset_batch)
        eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, eval_ids = batch

        # Run model.
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions)

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()
        pred = logits.data.max(1)[1].cpu() # get the index of the max log-probability

        eval_accumulate(model, data_manager, A, batch)
        A.add('class_correct', pred.eq(target).sum())
        A.add('class_total', target.size(0))

        # Optionally calculate transition loss/acc.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        total_tokens += sum([(nt+1)/2 for nt in eval_num_transitions_batch.reshape(-1)])

        if FLAGS.write_eval_report:
            reporter_args = [pred, target, eval_ids, output.data.cpu().numpy()]
            if hasattr(model, 'transition_loss'):
                transitions_per_example, _ = model.spinn.get_transitions_per_example(
                    style="preds" if FLAGS.eval_report_use_preds else "given")
                if model.use_sentence_pair:
                    batch_size = pred.size(0)
                    sent1_transitions = transitions_per_example[:batch_size]
                    sent2_transitions = transitions_per_example[batch_size:]
                    reporter_args.append(sent1_transitions)
                    reporter_args.append(sent2_transitions)
                else:
                    reporter_args.append(transitions_per_example)
            reporter.save_batch(*reporter_args)

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    end = time.time()
    total_time = end - start

    A.add('total_tokens', total_tokens)
    A.add('total_time', total_time)

    stats_args = eval_stats(model, A, step)
    stats_args['filename'] = filename

    logger.Log(eval_str.format(**stats_args))
    logger.Log(eval_extra_str.format(**stats_args))

    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".report")
        reporter.write_report(eval_report_path)

    eval_class_acc = stats_args['class_acc']
    eval_trans_acc = stats_args['transition_acc']

    if index == 0:
        eval_metrics(M, stats_args, step)

    return eval_class_acc, eval_trans_acc


def run():
    logger = afs_safe_logger.Logger(log_path(FLAGS))

    data_manager = get_data_manager(FLAGS.data_type)

    logger.Log("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Get Data and Embeddings
    vocabulary, initial_embeddings, _, eval_iterators = \
        load_data_and_embeddings(FLAGS, data_manager, logger, FLAGS.eval_data_path, FLAGS.eval_data_path)

    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(data_manager.LABEL_MAP)

    model, optimizer, trainer = init_model(FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager)

    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.load_experiment_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.load_experiment_name, best=True)

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
    recursively_set_device(optimizer.state_dict(), FLAGS.gpu)

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)

    # Do an evaluation-only run.
    eval_str = eval_format(model)
    logger.Log("Eval-Format: {}".format(eval_str))
    eval_extra_str = eval_extra_format(model)
    logger.Log("Eval-Extra-Format: {}".format(eval_extra_str))

    index = 0
    eval_set = eval_iterators[index]
    acc = evaluate(FLAGS, model, data_manager, eval_set, index, logger, step, vocabulary)


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS, load_log_flags=True)

    if len(FLAGS.eval_data_path.split(":")) > 1:
        raise Exception("The evaluate.py script only runs against one eval set. "
            "Please refrain from the ':' token in --eval_data_path")

    run()
