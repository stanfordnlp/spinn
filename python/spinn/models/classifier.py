"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m rembed.models.classifier --training_data_path bl-data/pbl_train.tsv \
       --eval_data_path bl-data/pbl_dev.tsv

SST sentiment (Demo only, model needs a full GloVe embeddings file to do well):
python -m rembed.models.classifier --data_type sst --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt --embedding_data_path rembed/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

SNLI entailment (Demo only, model needs a full GloVe embeddings file to do well):
python -m rembed.models.classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl \
       --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path rembed/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

Note: If you get an error starting with "TypeError: ('Wrong number of dimensions..." during development,
    there may already be a saved checkpoint in ckpt_path that matches the name of the model you're developing.
    Move or delete it as appropriate.
"""

from copy import copy
from functools import partial

from rembed import recurrences, util
from rembed.stack import ThinStack


def make_sentence_stack(cls, embedding_projection_network, compose_network,
                        vocab_size, seq_length, tokens, transitions,
                        num_classes, training_mode,
                        ground_truth_transitions_visible, vs, initial_embeddings,
                        project_embeddings, ss_mask_gen, ss_prob):
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

    return model


def _finalize_updates(updates, gradients):
    # Remove null parameter gradients.
    null_gradients = [k for k, v in gradients.iteritems() if v is None]
    logger.Log("The following parameters have null (disconnected) cost "
               "gradients and will not be trained: %s"
               % ", ".join(str(k) for k in null_gradients), logger.WARNING)
    for k in null_gradients:
        del gradients[key]

    updates += util.RMSprop(None, gradients.keys(), lr,
                            grads=gradients.values())
    return updates


def make_sentence_updates(model, cost, lr, X, y, transitions, training_mode,
                          ground_truth_transitions_visible):
    error_signal = T.grad(cost, model.final_representations)
    extra_cost_inputs = [y, training_mode, ground_truth_transitions_visible]

    model.make_backprop_scan(error_signal, extra_cost_inputs=extra_cost_inputs,
                             compute_embedding_gradients=train_embeddings)

    gradients = copy(model.gradients)
    if train_embeddings:
        gradients[model.embeddings] = model.embedding_gradients

    # Calculate gradients after stack fprop.
    vs = model._vs
    other_grads = {param: T.grad(cost, param)
                   for name, param in vs.trainable_vars
                   if name not in model._vars}
    gradients.update(other_grads)

    updates = model.scan_updates.items() + model.bscan_updates.items()
    updates += vs.nongradient_updates.items()
    return _finalize_updates(updates, gradients, lr)


def make_sentence_pair_updates(premise_model, hypothesis_model, cost, lr,
                               X, y, transitions, training_mode,
                               ground_truth_transitions_visible):
    premise_error_signal = T.grad(cost, premise_model.final_representations)
    hypothesis_error_signal = T.grad(cost, hypothesis_model.final_representations)

    extra_cost_inputs = [y, training_mode, ground_truth_transitions_visible,
                         X, transitions, premise_model.stack,
                         premise_model.aux_stacks]

    premise_model.make_backprop_scan(premise_error_signal,
                                     extra_cost_inputs=extra_cost_inputs,
                                     compute_embedding_gradients=False)

    hypothesis_model.make_backprop_scan(hypothesis_error_signal,
                                        extra_cost_inputs=extra_cost_inputs,
                                        compute_embedding_gradients=False)

    gradients = copy(premise_model.gradients)
    for k, h_grad in hypothesis_model.gradients.iteritems():
        if k in gradients:
            gradients[k] += h_grad
        else:
            gradients[k] = h_grad

    # Calculate gradients after stack fprop.
    vs = model._vs
    other_grads = {param: T.grad(cost, param)
                   for name, param in vs.trainable_vars
                   if name not in model._vars}
    gradients.update(other_grads)

    updates = util.merge_updates(
        premise_model.scan_updates + premise_model.bscan_updates,
        hypothesis_model.scan_updates + hypothesis_model.bscan_updates)
    updates += vs.nongradient_updates.iteritems()
    return _finalize_updates(updates, gradients, lr)


def make_sentence_pair_stacks(cls, embedding_projection_network, compose_network,
                              vocab_size, seq_length, tokens, transitions,
                              num_classes, training_mode,
                              ground_truth_transitions_visible, vs,
                              initial_embeddings, project_embeddings,
                              ss_mask_gen, ss_prob):

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
        use_attention=FLAGS.use_attention,
        name="premise")

    premise_stack_tops = premise_model.stack_tops if FLAGS.use_attention != "None" else None

    hypothesis_model = ThinStack(spec, recurrence, embedding_projection_network,
        training_mode, ground_truth_transitions_visible, vs,
        X=hypothesis_tokens,
        transitions=hypothesis_transitions,
        initial_embeddings=initial_embeddings,
        embedding_dropout_keep_rate=FLAGS.embedding_keep_rate,
        use_input_batch_norm=False,
        ss_mask_gen=ss_mask_gen,
        ss_prob=ss_prob,
        use_attention=FLAGS.use_attention,
        name="hypothesis")

    return premise_model, hypothesis_model


if __name__ == '__main__':
    FLAGS = core.define_flags()

    if FLAGS.data_type == "snli":
        make_stack_fn = make_sentence_pair_stacks
        make_updates_fn = make_sentence_pair_updates
    else:
        make_stack_fn = make_sentence_stack
        make_updates_fn = make_sentence_updates

    stack_cls = getattr(recurrences, FLAGS.model_type)
    make_stack_fn = partial(make_stack_fn, stack_cls)

    core.run(make_stack_fn, make_updates_fn,
             only_forward=FLAGS.expanded_eval_only_mode)
