"""Theano-based stack implementations."""

from functools import partial

import numpy as np
import theano

from theano import tensor as T
from rembed import cuda_util, util


def update_hard_stack(t, t_f, stack_t, push_value, merge_value, merge_queue_t,
                      merge_cursors_t, mask, batch_size, stack_shift, cursors_shift):
    """Compute the new value of the given hard stack.

    This performs stack pushes and pops in parallel, and somewhat wastefully.
    It accepts a precomputed merge result (in `merge_value`) and a precomputed
    push value `push_value` for all examples, and switches between the two
    outcomes based on the per-example value of `mask`.

    Args:
        stack_t: Current stack value
        stack_pushed: Helper stack structure, of same size as `stack_t`
        stack_merged: Helper stack structure, of same size as `stack_t`
        push_value: Batch of values to be pushed
        merge_value: Batch of merge results
        mask: Batch of booleans: 1 if merge, 0 if push
    """

    mask2 = mask.dimshuffle(0, "x")
    top_next = mask2 * merge_value + (1 - mask2) * push_value
    stack_next = cuda_util.AdvancedIncSubtensor1Floats(set_instead_of_inc=True, inplace=True)(
            stack_t, top_next, t_f * batch_size + stack_shift)

    cursors_next = merge_cursors_t + (mask * -1 + (1 - mask) * 1)
    queue_next = cuda_util.AdvancedIncSubtensor1Floats(set_instead_of_inc=True, inplace=True)(
            merge_queue_t, t_f, cursors_shift + cursors_next)

    return stack_next, queue_next, cursors_next


class HardStack(object):

    """
    Model 0/1/2 hard stack implementation.

    This model scans a sequence using a hard stack. It optionally predicts
    stack operations using an MLP, and can receive supervision on these
    predictions from some external parser which acts as the "ground truth"
    parser.

    Model 0: prediction_and_tracking_network=None, train_with_predicted_transitions=False
    Model 1: prediction_and_tracking_network=something, train_with_predicted_transitions=False
    Model 2: prediction_and_tracking_network=something, train_with_predicted_transitions=True
    """

    def __init__(self, model_dim, word_embedding_dim, batch_size, vocab_size, seq_length, compose_network,
                 embedding_projection_network, training_mode, ground_truth_transitions_visible, vs,
                 prediction_and_tracking_network=None,
                 predict_transitions=False,
                 train_with_predicted_transitions=False,
                 interpolate=False,
                 X=None,
                 transitions=None,
                 initial_embeddings=None,
                 make_test_fn=False,
                 use_input_batch_norm=True,
                 use_input_dropout=True,
                 embedding_dropout_keep_rate=1.0,
                 ss_mask_gen=None,
                 ss_prob=0.0,
                 use_tracking_lstm=False,
                 tracking_lstm_hidden_dim=8,
                 connect_tracking_comp=False,
                 context_sensitive_shift=False,
                 context_sensitive_use_relu=False):
        """
        Construct a HardStack.

        Args:
            model_dim: Dimensionality of token embeddings and stack values
            word_embedding_dim: dimension of the word embedding
            vocab_size: Number of unique tokens in vocabulary
            seq_length: Maximum sequence length which will be processed by this
              stack
            compose_network: Blocks-like function which accepts arguments
              `inp, inp_dim, outp_dim, vs, name` (see e.g. `util.Linear`).
              Given a Theano batch `inp` of dimension `batch_size * inp_dim`,
              returns a transformed Theano batch of dimension
              `batch_size * outp_dim`.
            embedding_projection_network: Same form as `compose_network`.
            training_mode: A Theano scalar indicating whether to act as a training model
              with dropout (1.0) or to act as an eval model with rescaling (0.0).
            ground_truth_transitions_visible: A Theano scalar. If set (1.0), allow the model access
              to ground truth transitions. This can be disabled at evaluation time to force Model 1
              (or 12SS) to evaluate in the Model 2 style with predicted transitions. Has no effect
              on Model 0.
            vs: VariableStore instance for parameter storage
            prediction_and_tracking_network: Blocks-like function which either maps values
              `3 * model_dim` to `action_dim` or uses the more complex TrackingUnit template.
            predict_transitions: If set, predict transitions. If not, the tracking LSTM may still
              be used for other purposes.
            train_with_predicted_transitions: If `True`, use the predictions from the model
              (rather than the ground-truth `transitions`) to perform stack
              operations
            interpolate: If True, use scheduled sampling while training
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
            transitions: Theano batch describing transition matrix, or `None`
              (in which case this instance will make its own batch variable).
            initial_embeddings: pretrained embeddings or None
            make_test_fn: If set, create a function to run a scan for testing.
            use_input_batch_norm: If True, use batch normalization
            use_input_dropout: If True, use dropout
            embedding_dropout_keep_rate: The keep rate for dropout on projected embeddings.
            ss_mask_gen: A theano random stream
            ss_prob: Scheduled sampling probability
            use_tracking_lstm: If True, LSTM will be used in the tracking unit
            tracking_lstm_hidden_dim: hidden state dimension of the tracking LSTM
            connect_tracking_comp: If True, the hidden state of tracking LSTM will be
                fed to the TreeLSTM in the composition unit
            context_sensitive_shift: If True, the hidden state of tracking LSTM and the embedding
                vector will be used to calculate the vector that will be pushed onto the stack
            context_sensitive_use_relu: If True, a ReLU layer will be used while doing context
                sensitive shift, otherwise a Linear layer will be used
        """

        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.stack_size = seq_length

        self._compose_network = compose_network
        self._embedding_projection_network = embedding_projection_network
        self._prediction_and_tracking_network = prediction_and_tracking_network
        self._predict_transitions = predict_transitions
        self.train_with_predicted_transitions = train_with_predicted_transitions

        self._vs = vs

        self.initial_embeddings = initial_embeddings

        self.training_mode = training_mode
        self.ground_truth_transitions_visible = ground_truth_transitions_visible
        self.embedding_dropout_keep_rate = embedding_dropout_keep_rate

        self.X = X
        self.transitions = transitions

        self.use_input_batch_norm = use_input_batch_norm
        self.use_input_dropout = use_input_dropout

        # Mask for scheduled sampling.
        self.ss_mask_gen = ss_mask_gen
        # Flag for scheduled sampling.
        self.interpolate = interpolate
        # Training step number.
        self.ss_prob = ss_prob
        # Connect tracking unit and composition unit.
        self.connect_tracking_comp = connect_tracking_comp
        assert (use_tracking_lstm or not connect_tracking_comp), \
            "Must use tracking LSTM if connecting tracking and composition units"
        self.context_sensitive_shift = context_sensitive_shift
        assert (use_tracking_lstm or not context_sensitive_shift), \
            "Must use tracking LSTM while doing context sensitive shift"
        self.context_sensitive_use_relu = context_sensitive_use_relu

        self._make_params()
        self._make_shared()
        self._make_inputs()
        self._make_scan()

        if make_test_fn:
            self.scan_fn = theano.function([self.X, self.transitions, self.training_mode,
                                            self.ground_truth_transitions_visible],
                                           self.stack, updates=self.scan_updates,
                                           accept_inplace=True,
                                           on_unused_input="warn")

    def _make_params(self):
        # Per-token embeddings.
        if self.initial_embeddings is not None:
            def EmbeddingInitializer(shape):
                return self.initial_embeddings
            self.embeddings = self._vs.add_param(
                    "embeddings", (self.vocab_size, self.word_embedding_dim),
                    initializer=EmbeddingInitializer,
                    trainable=False)
        else:
            self.embeddings = self._vs.add_param(
                "embeddings", (self.vocab_size, self.word_embedding_dim))

    def _make_shared(self):
        stack_init = np.zeros((self.stack_size * self.batch_size, self.model_dim), dtype=np.float32)
        self._stack_orig = theano.shared(stack_init, borrow=False, name="stack_orig")
        self.stack = theano.shared(stack_init, borrow=False, name="stack")

        cursors_init = np.zeros((self.batch_size,)).astype(np.float32)
        self._cursors_orig = theano.shared(cursors_init, borrow=False, name="cursors_orig")
        self.cursors = theano.shared(cursors_init, borrow=False, name="cursors")

        queue_init = np.zeros((self.batch_size * self.stack_size,)).astype(np.float32)
        self._queue_orig = theano.shared(queue_init, borrow=False, name="queue_orig")
        self.queue = theano.shared(queue_init, borrow=False, name="queue")

        zero_updates = {
                self.stack: self._stack_orig,
                self.cursors: self._cursors_orig,
                self.queue: self._queue_orig
        }
        self.zero = theano.function([], (), updates=zero_updates)

    def _make_inputs(self):
        self.X = self.X or T.imatrix("X")
        self.transitions = self.transitions or T.imatrix("transitions")

    def _step(self, t, t_f, transitions_t, transitions_t_f, ss_mask_gen_matrix_t,
              buffer_cur_t, tracking_hidden, buffer,
              ground_truth_transitions_visible):
        batch_size, _ = self.X.shape

        # Extract top buffer values.
        idxs = buffer_cur_t + self._buffer_shift

        if self.context_sensitive_shift:
            # Combine with the hidden state from previous unit.
            tracking_h_t = tracking_hidden[:, :self.tracking_lstm_hidden_dim]
            context_comb_input_t = T.concatenate([tracking_h_t, buffer[idxs]], axis=1)
            context_comb_input_dim = self.word_embedding_dim + self.tracking_lstm_hidden_dim
            comb_layer = util.ReLULayer if self.context_sensitive_use_relu else util.Linear
            buffer_top_t = comb_layer(context_comb_input_t, context_comb_input_dim, self.model_dim,
                                self._vs, name="context_comb_unit", use_bias=True,
                                initializer=util.HeKaimingInitializer())
        else:
            buffer_top_t = cuda_util.AdvancedSubtensor1Floats()(buffer, idxs)

        # Fetch top two stack elements.
        stack_1 = cuda_util.AdvancedSubtensor1Floats()(self.stack, (t - 1) * self.batch_size + self._stack_shift)
        # Get pointers into stack for second-to-top element.
        stack_2_ptrs = cuda_util.AdvancedSubtensor1Floats()(self.queue, self.cursors - 1.0 + self._queue_shift)
        stack_2_ptrs = stack_2_ptrs * batch_size + self._stack_shift
        # Retrieve second-to-top element.
        stack_2 = cuda_util.AdvancedSubtensor1Floats()(self.stack, stack_2_ptrs)

        if self._prediction_and_tracking_network is not None:
            # We are predicting our own stack operations.
            predict_inp = T.concatenate(
                [stack_1, stack_2, buffer_top_t], axis=1)

            if self.use_tracking_lstm:
                # Update the hidden state and obtain predicted actions.
                tracking_hidden, actions_t = self._prediction_and_tracking_network(
                    tracking_hidden, predict_inp, self.model_dim * 3,
                    self.tracking_lstm_hidden_dim, self._vs,
                    name="prediction_and_tracking")
            else:
                # Obtain predicted actions directly.
                actions_t = self._prediction_and_tracking_network(
                    predict_inp, self.model_dim * 3, util.NUM_TRANSITION_TYPES, self._vs,
                    name="prediction_and_tracking")

        if self.train_with_predicted_transitions:
            # Model 2 case.
            if self.interpolate:
                # Only use ground truth transitions if they are marked as visible to the model.
                effective_ss_mask_gen_matrix_t = ss_mask_gen_matrix_t * ground_truth_transitions_visible
                # Interpolate between truth and prediction using bernoulli RVs
                # generated prior to the step.
                mask = (transitions_t * effective_ss_mask_gen_matrix_t
                        + actions_t.argmax(axis=1) * (1 - effective_ss_mask_gen_matrix_t))
            else:
                # Use predicted actions to build a mask.
                mask = actions_t.argmax(axis=1)
        elif self._predict_transitions:
            # Use transitions provided from external parser when not masked out
            mask = (transitions_t * ground_truth_transitions_visible
                        + actions_t.argmax(axis=1) * (1 - ground_truth_transitions_visible))
        else:
            # Model 0 case
            mask = transitions_t_f

        # Now update the stack: first precompute merge results.
        merge_items = T.concatenate([stack_1, stack_2], axis=1)
        if self.connect_tracking_comp:
            tracking_h_t = tracking_hidden[:, :self.tracking_lstm_hidden_dim]
            merge_value = self._compose_network(merge_items, tracking_h_t, self.model_dim,
                self._vs, name="compose", external_state_dim=self.tracking_lstm_hidden_dim)
        else:
            merge_value = self._compose_network(merge_items, self.model_dim * 2, self.model_dim,
                self._vs, name="compose")

        # Compute new stack value.
        stack_next, merge_queue_next, merge_cursors_next = update_hard_stack(
            t, t_f, self.stack, buffer_top_t, merge_value, self.queue, self.cursors,
            mask, self.batch_size, self._stack_shift, self._cursors_shift)

        # Move buffer cursor as necessary. Since mask == 1 when merge, we
        # should increment each buffer cursor by 1 - mask.
        buffer_cur_next = buffer_cur_t + (1 - transitions_t_f)

        if self._predict_transitions:
            ret_val = buffer_cur_next, tracking_hidden, actions_t
        else:
            ret_val = buffer_cur_next, tracking_hidden, stack_2_ptrs

        if not self.interpolate:
            # Use ss_mask as a redundant return value.
            ret_val = (ss_mask_gen_matrix_t,) + ret_val

        updates = {
            self.stack: stack_next,
            self.queue: merge_queue_next,
            self.cursors: merge_cursors_next
        }

        return ret_val, updates

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_size = self.batch_size
        max_stack_size = stack_size = self.stack_size
        self.batch_range = batch_range = T.arange(batch_size, dtype="int32")

        self._queue_shift = T.cast(batch_range * self.seq_length,
                                   theano.config.floatX)
        self._buffer_shift = self._queue_shift
        self._cursors_shift = T.cast(batch_range * self.stack_size,
                                     theano.config.floatX)
        self._stack_shift = T.cast(batch_range, theano.config.floatX)

        # Look up all of the embeddings that will be used.
        raw_embeddings = self.embeddings[self.X]  # batch_size * seq_length * emb_dim

        if self.context_sensitive_shift:
            # Use the raw embedding vectors, they will be combined with the hidden state of
            # the tracking unit later
            buffer_t = raw_embeddings
            buffer_emb_dim = self.word_embedding_dim
        else:
            # Allocate a "buffer" stack initialized with projected embeddings,
            # and maintain a cursor in this buffer.
            buffer_t = self._embedding_projection_network(
                raw_embeddings, self.word_embedding_dim, self.model_dim, self._vs, name="project")
            if self.use_input_batch_norm:
                buffer_t = util.BatchNorm(buffer_t, self.model_dim, self._vs, "buffer",
                    self.training_mode, axes=[0, 1])
            if self.use_input_dropout:
                buffer_t = util.Dropout(buffer_t, self.embedding_dropout_keep_rate, self.training_mode)
            buffer_emb_dim = self.model_dim

        # Collapse buffer to (batch_size * buffer_size) * emb_dim for fast indexing.
        buffer_t = buffer_t.reshape((-1, buffer_emb_dim))

        buffer_cur_init = T.zeros((batch_size,), theano.config.floatX)

        DUMMY = T.zeros((2,)) # a dummy tensor used as a place-holder

        # Dimshuffle inputs to seq_len * batch_size for scanning
        transitions = self.transitions.dimshuffle(1, 0)
        transitions_f = T.cast(transitions, dtype=theano.config.floatX)

        # Initialize the hidden state for the tracking LSTM, if needed.
        if self.use_tracking_lstm:
            # TODO: Unify what 'dim' means with LSTM. Here, it's the dim of
            # each of h and c. For 'model_dim', it's the combined dimension
            # of the full hidden state (so h and c are each model_dim/2).
            hidden_init = T.zeros((batch_size, self.tracking_lstm_hidden_dim * 2))
        else:
            hidden_init = DUMMY

        # Set up the output list for scanning over _step().
        if self._predict_transitions:
            outputs_info = [stack_init, buffer_cur_init, hidden_init, None]
        else:
            outputs_info = [buffer_cur_init, hidden_init, None]

        # Prepare data to scan over.
        sequences = [T.arange(transitions.shape[0]),
                     T.arange(transitions.shape[0], dtype="float32"),
                     transitions, transitions_f]
        if self.interpolate:
            # Generate Bernoulli RVs to simulate scheduled sampling
            # if the interpolate flag is on.
            ss_mask_gen_matrix = self.ss_mask_gen.binomial(
                                transitions.shape, p=self.ss_prob)
            # Take in the RV sequence as input.
            sequences.append(ss_mask_gen_matrix)
        else:
            # Take in the RV sequqnce as a dummy output. This is
            # done to avaid defining another step function.
            outputs_info = [DUMMY] + outputs_info

        scan_ret, self.scan_updates = theano.scan(
                self._step,
                sequences=sequences,
                non_sequences=[buffer_t, self.ground_truth_transitions_visible],
                outputs_info=outputs_info)

        ret_shift = 0 if self.interpolate else 1
        self.final_buf = scan_ret[ret_shift + 0][-1]
        self.stack_2_ptrs = scan_ret[ret_shift + 2]
        self.buf_ptrs = scan_ret[ret_shift + 0]

        self.final_stack = self.scan_updates[self.stack]

        self.transitions_pred = None
        if self._predict_transitions:
            self.transitions_pred = scan_ret[-1].dimshuffle(1, 0, 2)


    def make_backprop_scan(self, error_signal, f_delta, grad_shapes):
        """
        Args:
            error_signal: Theano batch of batch_size * model_dim
        """

        if not hasattr(self, "stack_2_ptrs"):
            raise RuntimeError("self._make_scan (forward pass) must be defined "
                               "before self.make_backprop_scan is called")

        ## Backprop scan ##
        # defined for simple RNN case, where each merge is ([c1; c2] * W)

        stack_bwd_init = T.zeros((self.stack_size * self.batch_size, self.model_dim))
        stack_bwd_init = T.set_subtensor(stack_bwd_init[-self.batch_size:], error_signal)

        batch_size = self.batch_size
        batch_range = T.arange(batch_size)
        stack_shift = T.cast(batch_range, theano.config.floatX)
        buffer_shift = T.cast(batch_range * self.seq_length, theano.config.floatX)

        def step_b(# sequences
                   t_f, transitions_t_f, stack_2_ptrs_t, buffer_cur_t,
                   # outputs_info (inplace update is okay)
                   stack_bwd_t, dE,
                   # rest
                   *accum_and_non_sequences):
            accum_deltas, stack_final = accum_and_non_sequences[:-1], \
                accum_and_non_sequences[-1]

            err_prev = cuda_util.AdvancedSubtensor1Floats()(
                stack_bwd_t, t_f * batch_size + stack_shift)

            # Find the timesteps of the two elements involved in the potential
            # merge at this timestep.
            t_c1 = (t_f - 1.0) * batch_size + stack_shift
            t_c2 = stack_2_ptrs_t

            # Find the two elements involved in the merge.
            # batch_size * model_dim
            c1 = cuda_util.AdvancedSubtensor1Floats()(stack_final, t_c1)
            c2 = cuda_util.AdvancedSubtensor1Floats()(stack_final, t_c2)

            # Calculate deltas for this timestep.
            delta, d_compose = f_delta((c1, c2), (err_prev,))
            # TODO support multiple graph inputs
            delta = delta[0]

            # Calculate deltas of dE for each element.
            dE_push = err_prev
            buffer_ids_t = cuda_util.AdvancedSubtensor1Floats()(
                    id_buffer, buffer_cur_t + buffer_shift)

            # Calculate delta vectors d(cost)/d(stack_val) for preceding
            # timestep.
            # 2 * batch_size * model_dim
            err_c1 = delta[:, :self.model_dim]
            err_c2 = delta[:, self.model_dim:]

            ## Switch between two cases.
            # TODO: Record actual transitions (e.g. for model 1S and higher)
            # and repeat those here
            mask = transitions_t_f
            masks = [mask, mask.dimshuffle(0, "x"),
                     mask.dimshuffle(0, "x", "x")]

            # TODO: Is this at all efficient? (Bring back GPURowSwitch?)
            new_accum_deltas = []
            for accum_delta, delta in zip(accum_deltas, d_compose):
                assert accum_delta.ndim == delta.ndim - 1, delta.ndim
                mask_i = masks[delta.ndim - 1]
                # TODO: Is this at all efficient? (Bring back GPURowSwitch?)
                delta = (mask * delta).sum(axis=0)
                new_accum_deltas.append(accum_delta + delta)

            dE = T.inc_subtensor(dE[buffer_ids_t], (1. - masks[1]) * dE_push)

            # Update backward-pass stack structure.
            # For each example:
            #   Retrieve positions of potential merge elements (t_c1, t_c2)
            #   If we merged: backprop error signals to these positions
            #   If we pushed: leave these positions unchanged
            # DEV: Can't force inplace until we update Theano internals to
            # admit inplace updates on stack_bwd_t
            stack_bwd_next = cuda_util.AdvancedIncSubtensor1Floats()(#, inplace=True)(
                    stack_bwd_t, masks[1] * err_c1, t_c1)
            stack_bwd_next = cuda_util.AdvancedIncSubtensor1Floats()(#), inplace=True)(
                    stack_bwd_next, masks[1] * err_c2, t_c2)

            return [stack_bwd_next, dE] + new_accum_deltas

        # TODO: These should come from forward pass -- not fixed -- in model
        # 1S, etc.
        transitions_f = T.cast(self.transitions.dimshuffle(1, 0),
                               dtype=theano.config.floatX)

        ts_f = T.cast(T.arange(transitions_f.shape[0]), dtype=theano.config.floatX)

        # Representation of buffer using embedding indices rather than values
        id_buffer = self.X.flatten()
        # Build sequence of buffer pointers, where buf_ptrs[i] indicates the
        # buffer pointer values *before* computation at timestep *i* proceeds.
        # (This means we need to slice off the last actual buf_ptr output and
        # prepend a dummy.)
        buf_ptrs = T.concatenate([T.zeros((1, batch_size,)),
                                  self.buf_ptrs[:-1]], axis=0)

        outputs_info = [stack_bwd_init, T.zeros_like(self.embeddings)]
        outputs_info += [T.zeros(shape) for shape in grad_shapes]

        bscan_ret, _ = theano.scan(
                step_b,
                sequences=[ts_f, transitions_f, self.stack_2_ptrs, buf_ptrs],
                outputs_info=outputs_info,
                non_sequences=[self.final_stack],
                go_backwards=True)

        stack_bwd, dE = bscan_ret[:2]
        self.deltas = [deltas[-1] for deltas in bscan_ret[2:]]
        self.dE = dE[-1]


class Model0(HardStack):

    def __init__(self, *args, **kwargs):
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = partial(util.TrackingUnit, make_logits=False)
        else:
            kwargs["prediction_and_tracking_network"] = None

        kwargs["predict_transitions"] = False
        kwargs["train_with_predicted_transitions"] = False
        kwargs["interpolate"] = False
        super(Model0, self).__init__(*args, **kwargs)


class Model1(HardStack):

    def __init__(self, *args, **kwargs):
        # Set the tracking unit based on supplied tracking_lstm_hidden_dim.
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = util.TrackingUnit
        else:
            kwargs["prediction_and_tracking_network"] = util.Linear
        # Defaults to not using predictions while training and not using scheduled sampling.
        kwargs["predict_transitions"] = True
        kwargs["train_with_predicted_transitions"] = False
        kwargs["interpolate"] = False
        super(Model1, self).__init__(*args, **kwargs)


class Model2(HardStack):

    def __init__(self, *args, **kwargs):
        # Set the tracking unit based on supplied tracking_lstm_hidden_dim.
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = util.TrackingUnit
        else:
            kwargs["prediction_and_tracking_network"] = util.Linear
        # Defaults to using predictions while training and not using scheduled sampling.
        kwargs["predict_transitions"] = True
        kwargs["train_with_predicted_transitions"] = True
        kwargs["interpolate"] = False
        super(Model2, self).__init__(*args, **kwargs)


class Model2S(HardStack):

    def __init__(self, *args, **kwargs):
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = util.TrackingUnit
        else:
            kwargs["prediction_and_tracking_network"] = util.Linear
        # Use supplied settings and use scheduled sampling.
        kwargs["predict_transitions"] = True
        kwargs["train_with_predicted_transitions"] = True
        kwargs["interpolate"] = True
        super(Model2S, self).__init__(*args, **kwargs)
