"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m spinn.models.classifier --training_data_path bl-data/pbl_train.tsv \
       --eval_data_path bl-data/pbl_dev.tsv

SST sentiment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.classifier --data_type sst --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

SNLI entailment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl \
       --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

Note: If you get an error starting with "TypeError: ('Wrong number of dimensions..." during development,
    there may already be a saved checkpoint in ckpt_path that matches the name of the model you're developing.
    Move or delete it as appropriate.
"""

from functools import partial
import os
import pprint
import sys

import gflags
from theano import tensor as T
import theano
import numpy as np

from spinn import afs_safe_logger
from spinn import recurrences, util
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.stack import ThinStack

import spinn.plain_rnn


FLAGS = gflags.FLAGS


def build_sentence_model(cls, vocab_size, seq_length, tokens, transitions,
                         num_classes, training_mode, ground_truth_transitions_visible, vs,
                         initial_embeddings=None, project_embeddings=False, ss_mask_gen=None, ss_prob=0.0):
    """
    Construct a classifier which makes use of some hard-stack model.

    Args:
      cls: Hard stack class to use (from e.g. `spinn.stack`)
      vocab_size:
      seq_length: Length of each sequence provided to the stack model
      tokens: Theano batch (integer matrix), `batch_size * seq_length`
      transitions: Theano batch (integer matrix), `batch_size * seq_length`
      num_classes: Number of output classes
      training_mode: A Theano scalar indicating whether to act as a training model
        with dropout (1.0) or to act as an eval model with rescaling (0.0).
      ground_truth_transitions_visible: A Theano scalar. If set (1.0), allow the model access
        to ground truth transitions. This can be disabled at evaluation time to force Model 1
        (or 2S) to evaluate in the Model 2 style with predicted transitions. Has no effect on Model 0.
      vs: Variable store.
    """

    # Prepare layer which performs stack element composition.
    if cls is spinn.plain_rnn.RNN:
        compose_network = partial(util.LSTMLayer,
                                      initializer=util.HeKaimingInitializer())
        embedding_projection_network = None
    else:
        if FLAGS.lstm_composition:
            compose_network = partial(util.TreeLSTMLayer,
                                      initializer=util.HeKaimingInitializer())
        else:
            assert not FLAGS.connect_tracking_comp, "Can only connect tracking and composition unit while using TreeLSTM"
            compose_network = partial(util.ReLULayer,
                                      initializer=util.HeKaimingInitializer())

        if project_embeddings:
            embedding_projection_network = util.Linear
        else:
            assert FLAGS.word_embedding_dim == FLAGS.model_dim, \
                "word_embedding_dim must equal model_dim unless a projection layer is used."
            embedding_projection_network = util.IdentityLayer

    model_visible_dim = FLAGS.model_dim / 2 if FLAGS.lstm_composition else FLAGS.model_dim
    spec = util.ModelSpec(FLAGS.model_dim, FLAGS.word_embedding_dim,
                          FLAGS.batch_size, vocab_size, seq_length,
                          model_visible_dim=model_visible_dim)

    # TODO: Check non-Model0 support.
    recurrence = cls(spec, vs, compose_network,
                     use_context_sensitive_shift=FLAGS.context_sensitive_shift,
                     context_sensitive_use_relu=FLAGS.context_sensitive_use_relu,
                     use_tracking_lstm=FLAGS.use_tracking_lstm,
                     tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim)

    model = ThinStack(spec, recurrence, embedding_projection_network,
                      training_mode, ground_truth_transitions_visible, vs,
                      X=tokens,
                      transitions=transitions,
                      initial_embeddings=initial_embeddings,
                      embedding_dropout_keep_rate=FLAGS.embedding_keep_rate,
                      use_input_batch_norm=False,
                      ss_mask_gen=ss_mask_gen,
                      ss_prob=ss_prob)

    # Extract top element of final stack timestep.
    if FLAGS.lstm_composition:
        sentence_vector = model.sentence_embeddings[:, :FLAGS.model_dim / 2]
        sentence_vector_dim = FLAGS.model_dim / 2
    else:
        sentence_vector = model.sentence_embeddings
        sentence_vector_dim = FLAGS.model_dim

    sentence_vector = util.BatchNorm(sentence_vector, sentence_vector_dim, vs, "sentence_vector", training_mode)
    sentence_vector = util.Dropout(sentence_vector, FLAGS.semantic_classifier_keep_rate, training_mode)

    # Feed forward through a single output layer
    logits = util.Linear(
        sentence_vector, sentence_vector_dim, num_classes, vs,
        name="semantic_classifier", use_bias=True)

    def zero_fn():
        model.zero()

    return model, logits, zero_fn


def build_sentence_pair_model(cls, vocab_size, seq_length, tokens, transitions,
                     num_classes, training_mode, ground_truth_transitions_visible, vs,
                     initial_embeddings=None, project_embeddings=False, ss_mask_gen=None, ss_prob=0.0):
    """
    Construct a classifier which makes use of some hard-stack model.

    Args:
      cls: Hard stack class to use (from e.g. `spinn.stack`)
      vocab_size:
      seq_length: Length of each sequence provided to the stack model
      tokens: Theano batch (integer matrix), `batch_size * seq_length`
      transitions: Theano batch (integer matrix), `batch_size * seq_length`
      num_classes: Number of output classes
      training_mode: A Theano scalar indicating whether to act as a training model
        with dropout (1.0) or to act as an eval model with rescaling (0.0).
      ground_truth_transitions_visible: A Theano scalar. If set (1.0), allow the model access
        to ground truth transitions. This can be disabled at evaluation time to force Model 1
        (or 2S) to evaluate in the Model 2 style with predicted transitions. Has no effect on Model 0.
      vs: Variable store.
    """

    # Prepare layer which performs stack element composition.
    if cls is spinn.plain_rnn.RNN:
        compose_network = partial(util.LSTMLayer,
                                      initializer=util.HeKaimingInitializer())
        embedding_projection_network = None
    else:
        if FLAGS.lstm_composition:
            compose_network = partial(util.TreeLSTMLayer,
                                      initializer=util.HeKaimingInitializer())
        else:
            assert not FLAGS.connect_tracking_comp, "Can only connect tracking and composition unit while using TreeLSTM"
            compose_network = partial(util.ReLULayer,
                                      initializer=util.HeKaimingInitializer())

        if project_embeddings:
            embedding_projection_network = util.Linear
        else:
            assert FLAGS.word_embedding_dim == FLAGS.model_dim, \
                "word_embedding_dim must equal model_dim unless a projection layer is used."
            embedding_projection_network = util.IdentityLayer

    model_visible_dim = FLAGS.model_dim / 2 if FLAGS.lstm_composition else FLAGS.model_dim
    spec = util.ModelSpec(FLAGS.model_dim, FLAGS.word_embedding_dim,
                          FLAGS.batch_size, vocab_size, seq_length,
                          model_visible_dim=model_visible_dim)

    # Split the two sentences
    premise_tokens = tokens[:, :, 0]
    hypothesis_tokens = tokens[:, :, 1]

    premise_transitions = transitions[:, :, 0]
    hypothesis_transitions = transitions[:, :, 1]

    # TODO: Check non-Model0 support.
    recurrence = cls(spec, vs, compose_network,
                     use_context_sensitive_shift=FLAGS.context_sensitive_shift,
                     context_sensitive_use_relu=FLAGS.context_sensitive_use_relu,
                     use_tracking_lstm=FLAGS.use_tracking_lstm,
                     tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim)

    # Build two hard stack models which scan over input sequences.
    premise_model = ThinStack(spec, recurrence, embedding_projection_network,
        training_mode, ground_truth_transitions_visible, vs,
        X=premise_tokens,
        transitions=premise_transitions,
        initial_embeddings=initial_embeddings,
        embedding_dropout_keep_rate=FLAGS.embedding_keep_rate,
        use_input_batch_norm=False,
        ss_mask_gen=ss_mask_gen,
        ss_prob=ss_prob,
        name="premise")

    hypothesis_model = ThinStack(spec, recurrence, embedding_projection_network,
        training_mode, ground_truth_transitions_visible, vs,
        X=hypothesis_tokens,
        transitions=hypothesis_transitions,
        initial_embeddings=initial_embeddings,
        embedding_dropout_keep_rate=FLAGS.embedding_keep_rate,
        use_input_batch_norm=False,
        ss_mask_gen=ss_mask_gen,
        ss_prob=ss_prob,
        name="hypothesis")

    # Create standard MLP features
    mlp_input = T.concatenate([premise_vector, hypothesis_vector], axis=1)
    mlp_input_dim = 2 * sentence_vector_dim

    if FLAGS.use_difference_feature:
        mlp_input = T.concatenate([mlp_input, premise_vector - hypothesis_vector], axis=1)
        mlp_input_dim += sentence_vector_dim

    if FLAGS.use_product_feature:
        mlp_input = T.concatenate([mlp_input, premise_vector * hypothesis_vector], axis=1)
        mlp_input_dim += sentence_vector_dim

    mlp_input = util.BatchNorm(mlp_input, mlp_input_dim, vs, "sentence_vectors", training_mode)
    mlp_input = util.Dropout(mlp_input, FLAGS.semantic_classifier_keep_rate, training_mode)

    # Apply a combining MLP
    prev_features = mlp_input
    prev_features_dim = mlp_input_dim
    for layer in range(FLAGS.num_sentence_pair_combination_layers):
        prev_features = util.ReLULayer(prev_features, prev_features_dim, FLAGS.sentence_pair_combination_layer_dim, vs,
            name="combining_mlp/" + str(layer),
            initializer=util.HeKaimingInitializer())
        prev_features_dim = FLAGS.sentence_pair_combination_layer_dim

        prev_features = util.BatchNorm(prev_features, prev_features_dim, vs, "combining_mlp/" + str(layer), training_mode)
        prev_features = util.Dropout(prev_features, FLAGS.semantic_classifier_keep_rate, training_mode)

    # Feed forward through a single output layer
    logits = util.Linear(
        prev_features, prev_features_dim, num_classes, vs,
        name="semantic_classifier", use_bias=True)

    def zero_fn():
        premise_model.zero()
        hypothesis_model.zero()

    return premise_model, hypothesis_model, logits, zero_fn


def build_cost(logits, targets):
    """
    Build a classification cost function.
    """
    # Clip gradients coming from the cost function.
    logits = theano.gradient.grad_clip(
        logits, -1. * FLAGS.clipping_max_value, FLAGS.clipping_max_value)

    predicted_dist = T.nnet.softmax(logits)

    costs = T.nnet.categorical_crossentropy(predicted_dist, targets)
    cost = costs.mean()

    pred = T.argmax(logits, axis=1)
    acc = 1. - T.mean(T.cast(T.neq(pred, targets), theano.config.floatX))

    return cost, acc


def build_transition_cost(logits, targets, num_transitions):
    """
    Build a parse action prediction cost function.
    """

    # swap seq_length dimension to front so that we can scan per timestep
    logits = T.swapaxes(logits, 0, 1)
    targets = targets.T

    def cost_t(logits, tgt, num_transitions):
        # TODO(jongauthier): Taper down xent cost as we proceed through
        # sequence?
        predicted_dist = T.nnet.softmax(logits)
        cost = T.nnet.categorical_crossentropy(predicted_dist, tgt)

        pred = T.argmax(logits, axis=1)
        error = T.neq(pred, tgt)
        return cost, error

    results, _ = theano.scan(cost_t, [logits, targets], non_sequences=[num_transitions])
    costs, errors = results

    # Create a mask that selects only transitions that involve real data.
    unrolling_length = T.shape(costs)[0]
    padding = unrolling_length - num_transitions
    padding = T.reshape(padding, (1, -1))
    rng = T.arange(unrolling_length) + 1
    rng = T.reshape(rng, (-1, 1))
    mask = T.gt(rng, padding)

    # Compute acc using the mask
    acc = 1.0 - (T.sum(errors * mask, dtype=theano.config.floatX)
                 / T.sum(num_transitions, dtype=theano.config.floatX))

    # Compute cost directly, since we *do* want a cost incentive to get the padding
    # transitions right.
    cost = T.mean(costs)
    return cost, acc


def evaluate(eval_fn, eval_set, logger, step, zero_fn):
    # Evaluate
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    for (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch) in eval_set[1]:
        acc_value, action_acc_value = eval_fn(
            eval_X_batch, eval_transitions_batch,
            eval_y_batch, eval_num_transitions_batch, 0.0,  # Eval mode: Don't apply dropout.
            int(FLAGS.allow_gt_transitions_in_eval),  # Allow GT transitions to be used according to flag.
            float(FLAGS.allow_gt_transitions_in_eval))  # If flag not set, used scheduled sampling
                                                        # p(ground truth) = 0.0,
                                                        # else SS p(ground truth) = 1.0
        acc_accum += acc_value
        action_acc_accum += action_acc_value
        eval_batches += 1.0

        # Zero out all auxiliary variables.
        zero_fn()
    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
              (step, acc_accum / eval_batches, action_acc_accum / eval_batches, eval_set[0]))
    return acc_accum / eval_batches


def evaluate_expanded(eval_fn, eval_set, eval_path, logger, step, sentence_pair_data, ind_to_word, zero_fn):
    """
    Write the  gold parses and predicted parses in the files <eval_out_path>.gld and <eval_out_path>.tst
    respectively. These files can be given as inputs to Evalb to evaluate parsing performance -

        evalb -p evalb_spinn.prm <eval_out_path>.gld  <eval_out_path>.tst
    """
    # TODO: Prune out redundant code, make usable on Model0 as well.
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    eval_gold_path = eval_path + ".gld"
    eval_out_path = eval_path + ".tst"
    eval_lbl_path = eval_path + ".lbl"
    with open(eval_gold_path, "w") as eval_gold, open(eval_out_path, "w") as eval_out:
        if FLAGS.write_predicted_label:
            label_out = open(eval_lbl_path, "w")
        if sentence_pair_data:
            for (eval_X_batch, eval_transitions_batch, eval_y_batch,
                    eval_num_transitions_batch) in eval_set[1]:
                acc_value, action_acc_value, sem_logit_values, logits_pred_hyp, logits_pred_prem = eval_fn(
                    eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch,
                    0.0,  # Eval mode: Don't apply dropout.
                    int(FLAGS.allow_gt_transitions_in_eval),  # Allow GT transitions to be used according to flag.
                    float(FLAGS.allow_gt_transitions_in_eval)) # adjust visibility of GT

                acc_accum += acc_value
                action_acc_accum += action_acc_value
                eval_batches += 1.0

                # Zero out all auxiliary variables.
                zero_fn()

                # write each predicted transition to file
                for orig_transitions, pred_logit_hyp, pred_logit_prem, tokens, true_class, example_sem_logits \
                        in zip(eval_transitions_batch, logits_pred_hyp,
                               logits_pred_prem, eval_X_batch, eval_y_batch, sem_logit_values):
                    orig_hyp_transitions, orig_prem_transitions = orig_transitions.T
                    hyp_tokens, prem_tokens = tokens.T
                    hyp_words = [ind_to_word[t] for t in hyp_tokens]
                    prem_words = [ind_to_word[t] for t in prem_tokens]
                    eval_gold.write(util.TransitionsToParse(orig_hyp_transitions, hyp_words) + "\n")
                    eval_out.write(util.TransitionsToParse(pred_logit_hyp.argmax(axis=1), hyp_words) + "\n")
                    eval_gold.write(util.TransitionsToParse(orig_prem_transitions, prem_words) + "\n")
                    eval_out.write(util.TransitionsToParse(pred_logit_prem.argmax(axis=1), prem_words) + "\n")

                    predicted_class = np.argmax(example_sem_logits)
                    exp_logit_values = np.exp(example_sem_logits)
                    class_probs = exp_logit_values / np.sum(exp_logit_values)
                    class_probs_repr = "\t".join(map(lambda p : "%.3f" % (p,), class_probs))
                    if FLAGS.write_predicted_label:
                        label_out.write(str(true_class == predicted_class) + "\t" + str(true_class)
                                  + "\t" + str(predicted_class) + "\t" + class_probs_repr + "\n")
        else:
            for (eval_X_batch, eval_transitions_batch, eval_y_batch,
                 eval_num_transitions_batch) in eval_set[1]:
                acc_value, action_acc_value, sem_logit_values, logits_pred = eval_fn(
                    eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch,
                    0.0,  # Eval mode: Don't apply dropout.
                    int(FLAGS.allow_gt_transitions_in_eval),  # Allow GT transitions to be used according to flag.
                    float(FLAGS.allow_gt_transitions_in_eval)) # adjust visibility of GT

                acc_accum += acc_value
                action_acc_accum += action_acc_value
                eval_batches += 1.0

                # write each predicted transition to file
                for orig_transitions, pred_logit, tokens, true_class, example_sem_logits \
                    in zip(eval_transitions_batch, logits_pred, eval_X_batch, eval_y_batch, sem_logit_values):
                    words = [ind_to_word[t] for t in tokens]
                    eval_gold.write(util.TransitionsToParse(orig_transitions, words) + "\n")
                    eval_out.write(util.TransitionsToParse(pred_logit.argmax(axis=1), words) + "\n")

                    predicted_class = np.argmax(example_sem_logits)
                    exp_logit_values = np.exp(example_sem_logits)
                    class_probs = exp_logit_values / np.sum(exp_logit_values)
                    class_probs_repr = "\t".join(map(lambda p : "%.3f" % (p,), class_probs))
                    if FLAGS.write_predicted_label:
                        label_out.write(str(true_class == predicted_class) + "\t" + str(true_class)
                                    + "\t" + str(predicted_class) + "\t" + class_probs_repr + "\n")

    logger.Log("Written gold parses in %s" % (eval_gold_path))
    logger.Log("Written predicted parses in %s" % (eval_out_path))
    if FLAGS.write_predicted_label:
        logger.Log("Written predicted labels in %s" % (eval_lbl_path))
        label_out.close()
    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
               (step, acc_accum / eval_batches, action_acc_accum / eval_batches, eval_set[0]))


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data
    else:
        logger.Log("Bad data type.")
        return

    if FLAGS.model_type != "Model0":
        raise NotImplementedError("Only basic model 0 (SPINN-PI, SPINN-PI-NT) "
                                  "is supported in the thin-stack "
                                  "implementation.")

    pp = pprint.PrettyPrinter(indent=4)
    logger.Log("Flag values:\n" + pp.pformat(FLAGS.FlagValuesDict()))

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path)

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            eval_data, _ = data_manager.load_data(eval_filename)
            raw_eval_sets.append((eval_filename, eval_data))

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
        initial_embeddings = util.LoadEmbeddingsFromASCII(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size)

    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        e_X, e_transitions, e_y, e_num_transitions = util.PreprocessDataset(
            raw_eval_set, vocabulary, FLAGS.seq_length, data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        eval_iterators.append((filename,
            util.MakeEvalIterator((e_X, e_transitions, e_y, e_num_transitions), FLAGS.batch_size)))

    # Set up the placeholders.
    y = T.vector("y", dtype="int32")
    lr = T.scalar("lr")
    training_mode = T.scalar("training_mode")  # 1: Training with dropout, 0: Eval
    ground_truth_transitions_visible = T.scalar("ground_truth_transitions_visible", dtype="int32")

    logger.Log("Building model.")
    vs = util.VariableStore(
        default_initializer=util.UniformInitializer(FLAGS.init_range), logger=logger)

    if FLAGS.model_type == "RNN":
        model_cls = spinn.plain_rnn.RNN
    else:
        model_cls = getattr(recurrences, FLAGS.model_type)

    # Generator of mask for scheduled sampling
    numpy_random = np.random.RandomState(1234)
    ss_mask_gen = T.shared_randomstreams.RandomStreams(numpy_random.randint(999999))

    # Training step number
    ss_prob = T.scalar("ss_prob")

    if data_manager.SENTENCE_PAIR_DATA:
        X = T.itensor3("X")
        transitions = T.itensor3("transitions")
        num_transitions = T.imatrix("num_transitions")

        premise_model, hypothesis_model, logits, zero_fn = build_sentence_pair_model(
            model_cls, len(vocabulary), FLAGS.seq_length,
            X, transitions, len(data_manager.LABEL_MAP), training_mode, ground_truth_transitions_visible, vs,
            initial_embeddings=initial_embeddings, project_embeddings=(not train_embeddings),
            ss_mask_gen=ss_mask_gen,
            ss_prob=ss_prob)
        premise_stack_top = premise_model.sentence_embeddings
        hypothesis_stack_top = hypothesis_model.sentence_embeddings
        predicted_premise_transitions = premise_model.transitions_pred
        predicted_hypothesis_transitions = hypothesis_model.transitions_pred
    else:
        X = T.matrix("X", dtype="int32")
        transitions = T.imatrix("transitions")
        num_transitions = T.vector("num_transitions", dtype="int32")

        model, logits, zero_fn = build_sentence_model(
            model_cls, len(vocabulary), FLAGS.seq_length,
            X, transitions, len(data_manager.LABEL_MAP), training_mode, ground_truth_transitions_visible, vs,
            initial_embeddings=initial_embeddings, project_embeddings=(not train_embeddings),
            ss_mask_gen=ss_mask_gen,
            ss_prob=ss_prob)
        stack_top = model.sentence_embeddings
        predicted_transitions = model.transitions_pred

    xent_cost, acc = build_cost(logits, y)

    # Set up L2 regularization.
    l2_cost = 0.0
    for var in vs.trainable_vars:
        l2_cost += FLAGS.l2_lambda * T.sum(T.sqr(vs.vars[var]))

    # Compute cross-entropy cost on action predictions.
    if (not data_manager.SENTENCE_PAIR_DATA) and predicted_transitions is not None:
        transition_cost, action_acc = build_transition_cost(predicted_transitions, transitions, num_transitions)
    elif data_manager.SENTENCE_PAIR_DATA and predicted_hypothesis_transitions is not None:
        p_transition_cost, p_action_acc = build_transition_cost(predicted_premise_transitions, transitions[:, :, 0],
            num_transitions[:, 0])
        h_transition_cost, h_action_acc = build_transition_cost(predicted_hypothesis_transitions, transitions[:, :, 1],
            num_transitions[:, 1])
        transition_cost = p_transition_cost + h_transition_cost
        action_acc = (p_action_acc + h_action_acc) / 2.0  # TODO(SB): Average over transitions, not words.
    else:
        transition_cost = T.constant(0.0)
        action_acc = T.constant(0.0)
    transition_cost = transition_cost * FLAGS.transition_cost_scale

    total_cost = xent_cost + l2_cost + transition_cost

    if ".ckpt" in FLAGS.ckpt_path:
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + ".ckpt")
    if os.path.isfile(checkpoint_path):
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = vs.load_checkpoint(checkpoint_path, num_extra_vars=2,
                                                  skip_saved_unsavables=FLAGS.skip_saved_unsavables)
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    # Do an evaluation-only run.
    if only_forward:
        if FLAGS.eval_output_paths:
            eval_output_paths = FLAGS.eval_output_paths.strip().split(":")
            assert len(eval_output_paths) == len(eval_iterators), "Invalid no. of output paths."
        else:
            eval_output_paths = [FLAGS.experiment_name + "-" + os.path.split(eval_set[0])[1] + "-parse"
                                  for eval_set in eval_iterators]

        # Load model from checkpoint.
        logger.Log("Checkpointed model was trained for %d steps." % (step,))

        # Generate function for forward pass.
        logger.Log("Building forward pass.")
        if data_manager.SENTENCE_PAIR_DATA:
            eval_fn = theano.function(
                [X, transitions, y, num_transitions, training_mode, ground_truth_transitions_visible, ss_prob],
                [acc, action_acc, logits, predicted_hypothesis_transitions, predicted_premise_transitions],
                on_unused_input='warn',
                allow_input_downcast=True)
        else:
            eval_fn = theano.function(
                [X, transitions, y, num_transitions, training_mode, ground_truth_transitions_visible, ss_prob],
                [acc, action_acc, logits, predicted_transitions],
                on_unused_input='warn',
                allow_input_downcast=True)

        # Generate the inverse vocabulary lookup table.
        ind_to_word = {v : k for k, v in vocabulary.iteritems()}

        # Do a forward pass and write the output to disk.
        for eval_set, eval_out_path in zip(eval_iterators, eval_output_paths):
            logger.Log("Writing eval output for %s." % (eval_set[0],))
            evaluate_expanded(eval_fn, eval_set, eval_out_path, logger, step,
                              data_manager.SENTENCE_PAIR_DATA, ind_to_word, zero_fn)
    else:
        # Train
        extra_cost_inputs = [y, training_mode, ground_truth_transitions_visible]
        if data_manager.SENTENCE_PAIR_DATA:
            # The two models use slices of the original data.
            # Pass the original data as a non-sequence input as well.
            extra_cost_inputs += [X, transitions]

            premise_error_signal = T.grad(total_cost, premise_stack_top)
            premise_model.make_backprop_scan(premise_error_signal,
                                             extra_cost_inputs=extra_cost_inputs,
                                             compute_embedding_gradients=False)

            extra_cost_inputs += [premise_model.stack] + premise_model.aux_stacks
            hypothesis_error_signal = T.grad(total_cost, hypothesis_stack_top)
            hypothesis_model.make_backprop_scan(hypothesis_error_signal,
                                                extra_cost_inputs=extra_cost_inputs,
                                                compute_embedding_gradients=False)

            gradients = premise_model.gradients
            hypothesis_gradients = hypothesis_model.gradients
            for key in hypothesis_gradients:
                if key in gradients:
                    gradients[key] += hypothesis_gradients[key]
                else:
                    gradients[key] = hypothesis_gradients[key]

            new_values = util.merge_updates(
                premise_model.scan_updates + premise_model.bscan_updates,
                hypothesis_model.scan_updates + hypothesis_model.bscan_updates).items()
            other_params = set(vs.trainable_vars.keys()) - premise_model._vars
            other_params -= hypothesis_model._vars
        else:
            error_signal = T.grad(total_cost, stack_top)
            model.make_backprop_scan(error_signal,
                                     extra_cost_inputs=extra_cost_inputs,
                                     compute_embedding_gradients=train_embeddings)
            if train_embeddings:
                model.gradients[model.embeddings] = model.embedding_gradients
            gradients = model.gradients

            new_values = model.scan_updates.items() + model.bscan_updates.items()
            other_params = set(vs.trainable_vars.keys()) - model._vars

        # Remove null stack parameter gradients.
        null_gradients = set()
        for key, val in gradients.iteritems():
            if val is None:
                null_gradients.add(key)
        if null_gradients:
            logger.Log("The following parameters have null (disconnected) cost "
                    "gradients and will not be trained: %s"
                    % ", ".join(str(k) for k in null_gradients), logger.WARNING)
        for key in null_gradients:
            del gradients[key]

        # Calculate gradients for items before/after stack fprop.
        other_params = [vs.vars[param] for param in other_params]
        other_grads = T.grad(total_cost, wrt=other_params)
        gradients.update(zip(other_params, other_grads))

        new_values += util.RMSprop(total_cost, gradients.keys(), lr,
                                   grads=gradients.values())
        new_values += [(key, vs.nongradient_updates[key]) for key in vs.nongradient_updates]

        # Create training and eval functions.
        # Unused variable warnings are supressed so that num_transitions can be passed in when training Model 0,
        # which ignores it. This yields more readable code that is very slightly slower.
        logger.Log("Building update function.")
        update_fn = theano.function(
            [X, transitions, y, num_transitions, lr, training_mode, ground_truth_transitions_visible, ss_prob],
            [total_cost, xent_cost, transition_cost, action_acc, l2_cost, acc],
            updates=new_values,
            on_unused_input='warn',
            allow_input_downcast=True)
        logger.Log("Building eval function.")
        eval_fn = theano.function([X, transitions, y, num_transitions, training_mode, ground_truth_transitions_visible, ss_prob], [acc, action_acc],
            on_unused_input='warn',
            allow_input_downcast=True)
        logger.Log("Training.")

        # Main training loop.
        for step in range(step, FLAGS.training_steps):
            if step % FLAGS.eval_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(eval_fn, eval_set, logger, step, zero_fn)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > 1000:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        vs.save_checkpoint(checkpoint_path + "_best", extra_vars=[step, best_dev_error])

            X_batch, transitions_batch, y_batch, num_transitions_batch = training_data_iter.next()
            # HACK: Drop training batches which aren't well-sized. (Will only
            # trigger for the final batch in a dataset.)
            if X_batch.shape[0] != FLAGS.batch_size:
                continue

            learning_rate = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))
            ret = update_fn(X_batch, transitions_batch, y_batch, num_transitions_batch,
                            learning_rate, 1.0, 1.0, np.exp(step*np.log(FLAGS.scheduled_sampling_exponent_base)))
            total_cost_val, xent_cost_val, transition_cost_val, action_acc_val, l2_cost_val, acc_val = ret

            if step % FLAGS.statistics_interval_steps == 0:
                logger.Log(
                    "Step: %i\tAcc: %f\t%f\tCost: %5f %5f %5f %5f"
                    % (step, acc_val, action_acc_val, total_cost_val, xent_cost_val, transition_cost_val,
                       l2_cost_val))

            if step % FLAGS.ckpt_interval_steps == 0 and step > 0:
                vs.save_checkpoint(checkpoint_path, extra_vars=[step, best_dev_error])

            # Zero out all auxiliary variables.
            zero_fn()


if __name__ == '__main__':
    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "experiment", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "snli"],
        "Which data handler and classifier to use.")

    # Where to store checkpoints
    gflags.DEFINE_string("ckpt_path", ".", "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("log_path", ".", "A directory in which to write logs.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", 30, "")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "Model0",
                       ["RNN", "Model0", "Model1", "Model2", "Model2S"],
                       "")
    # TODO(JG): Either add support for Models 1/2 or remove them as options.
    gflags.DEFINE_boolean("allow_gt_transitions_in_eval", False,
        "Whether to use ground truth transitions in evaluation when appropriate "
        "(i.e., in Model 1 and Model 2S.)")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")

    gflags.DEFINE_integer("tracking_lstm_hidden_dim", 4, "")
    gflags.DEFINE_boolean("use_tracking_lstm", True,
                          "Whether to use LSTM in the tracking unit")
    gflags.DEFINE_boolean("context_sensitive_shift", False,
        "Use LSTM hidden state and word embedding to determine the vector to be pushed")
    gflags.DEFINE_boolean("context_sensitive_use_relu", False,
        "Use ReLU Layer to combine embedding and tracking unit hidden state")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.5,
        "Used for dropout in the semantic task classifier.")
    gflags.DEFINE_float("embedding_keep_rate", 0.5,
        "Used for dropout on transformed embeddings.")
    gflags.DEFINE_boolean("lstm_composition", True, "")
    # gflags.DEFINE_integer("num_composition_layers", 1, "")
    gflags.DEFINE_integer("num_sentence_pair_combination_layers", 2, "")
    gflags.DEFINE_integer("sentence_pair_combination_layer_dim", 1024, "")
    gflags.DEFINE_float("scheduled_sampling_exponent_base", 0.99,
        "Used for scheduled sampling, with probability of Model 1 over Model 2 being base^#training_steps")
    gflags.DEFINE_boolean("use_difference_feature", True,
        "Supply the sentence pair classifier with sentence difference features.")
    gflags.DEFINE_boolean("use_product_feature", True,
        "Supply the sentence pair classifier with sentence product features.")
    gflags.DEFINE_boolean("connect_tracking_comp", True,
        "Connect tracking unit and composition unit. Can only be true if using LSTM in both units.")

    # Optimization settings.
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in RMSProp.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in RMSProp.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")
    gflags.DEFINE_float("transition_cost_scale", 1.0, "Multiplied by the transition cost.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")

    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_string("eval_output_paths", None,
        "Used when expanded_eval_only_mode is set. The number of supplied paths should be same"
        "as the number of eval sets.")
    gflags.DEFINE_boolean("write_predicted_label", False,
        "Write the predicted labels in a <eval_output_name>.lbl file.")
    gflags.DEFINE_boolean("skip_saved_unsavables", False,
        "Assume that variables marked as not savable will appear in checkpoints anyway, and "
        "skip them when loading. This should be used only when loading old checkpoints.")

    # Parse command line flags.
    FLAGS(sys.argv)

    run(only_forward=FLAGS.expanded_eval_only_mode)
