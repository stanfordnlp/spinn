"""Theano-based stack implementations."""


import tensorflow as tf

from rembed import util


def update_hard_stack_old(stack_t, stack_pushed, stack_merged, push_value,
                      merge_value, mask):
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

    # Build two copies of the stack batch: one where every stack has received
    # a push op, and one where every stack has received a merge op.
    #
    # Copy 1: Push.
    stack_pushed = T.set_subtensor(stack_pushed[:, 0], push_value)
    stack_pushed = T.set_subtensor(stack_pushed[:, 1:], stack_t[:, :-1])

    # Copy 2: Merge.
    stack_merged = T.set_subtensor(stack_merged[:, 0], merge_value)
    stack_merged = T.set_subtensor(stack_merged[:, 1:-1], stack_t[:, 2:])

    # Make sure mask broadcasts over all dimensions after the first.
    mask = mask.dimshuffle(0, "x", "x")
    mask = T.cast(mask, dtype=theano.config.floatX)
    stack_next = mask * stack_merged + (1. - mask) * stack_pushed

    return stack_next


def update_hard_stack(t, stack_t, push_value, merge_value,
                      queue_t, cursors_t, mask,
                      batch_size, stack_size, batch_range):

    mask = tf.expand_dims(mask, 1)
    mask_float = tf.to_float(mask)
    top_next = mask_float * merge_value + (1 - mask_float) * push_value
    stack_next = tf.scatter_update(
        stack_t, t * batch_size + batch_range, top_next)

    cursors_next = cursors_t + tf.squeeze(mask * -1 + (1 - mask) * 1)
    queue_next = tf.scatter_update(
        queue_t, batch_range * stack_size + cursors_next,
        tf.fill((batch_size,), t))

    return stack_next, queue_next, cursors_next


class HardStack(object):

    """
    Model 0/1/2 hard stack implementation.

    This model scans a sequence using a hard stack. It optionally predicts
    stack operations using an MLP, and can receive supervision on these
    predictions from some external parser which acts as the "ground truth"
    parser.

    Model 0: predict_network=None, use_predictions=False
    Model 1: predict_network=something, use_predictions=False
    Model 2: predict_network=something, use_predictions=True
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, batch_size, seq_length, compose_network,
                 embedding_projection_network, apply_dropout, vs, predict_network=None,
                 use_predictions=False, X=None, transitions=None, initial_embeddings=None,
                 make_test_fn=False, embedding_dropout_keep_rate=1.0):
        """
        Construct a HardStack.

        Args:
            model_dim: Dimensionality of token embeddings and stack values
            vocab_size: Number of unique tokens in vocabulary
            seq_length: Maximum sequence length which will be processed by this
              stack
            compose_network: Blocks-like function which accepts arguments
              `inp, inp_dim, outp_dim, vs, name` (see e.g. `util.Linear`).
              Given a Theano batch `inp` of dimension `batch_size * inp_dim`,
              returns a transformed Theano batch of dimension
              `batch_size * outp_dim`.
            embedding_projection_network: Same form as `compose_network`.
            apply_dropout: A Theano scalar indicating whether to apply dropout (1.0)
              or eval-mode rescaling (0.0).
            vs: VariableStore instance for parameter storage
            predict_network: Blocks-like function which maps values
              `3 * model_dim` to `action_dim`
            use_predictions: If `True`, use the predictions from the model
              (rather than the ground-truth `transitions`) to perform stack
              operations
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
            transitions: Theano batch describing transition matrix, or `None`
              (in which case this instance will make its own batch variable).
            make_test_fn: If set, create a function to run a scan for testing.
            embedding_dropout_keep_rate: The keep rate for dropout on projected
              embeddings.
        """

        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self._compose_network = compose_network
        self._embedding_projection_network = embedding_projection_network
        self._predict_network = predict_network
        self.use_predictions = use_predictions

        self._vs = vs

        self.initial_embeddings = initial_embeddings

        self.apply_dropout = apply_dropout
        self.embedding_dropout_keep_rate = embedding_dropout_keep_rate

        self.X = X
        self.transitions = transitions

        self._make_params()
        self._make_helpers()
        self._make_inputs()
        self._make_scan()

    def _make_params(self):
        # Per-token embeddings.
        if self.initial_embeddings is not None:
            def EmbeddingInitializer(shape):
                return self.initial_embeddings
            self.embeddings = self._vs.add_param(
                "embeddings", (self.vocab_size, self.word_embedding_dim), initializer=EmbeddingInitializer)
        else:
            self.embeddings = self._vs.add_param(
                "embeddings", (self.vocab_size, self.word_embedding_dim))

    def _make_helpers(self):
        self._stack = tf.Variable(
                tf.zeros((self.seq_length * self.batch_size, self.model_dim)),
                name="stack")
        self._queue = tf.Variable(
                tf.zeros((self.batch_size * self.seq_length,), dtype=tf.int32),
                name="queue")
        self._cursors = tf.Variable(tf.fill((self.batch_size,), -1),
                                    name="cursors")

    def zero(self, sess):
        """Prepare helpers for a feedforward/backprop."""
        sess.run(tf.initialize_variables([self._stack, self._queue, self._cursors]))

    def _make_inputs(self):
        self.X = self.X or T.imatrix("X")
        self.transitions = self.transitions or T.imatrix("transitions")

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_range = tf.range(0, self.batch_size)
        batch_range.set_shape((self.batch_size,))

        # Look up all of the embeddings that will be used.
        raw_embeddings = tf.nn.embedding_lookup(self.embeddings, self.X) # batch_size * seq_length * emb_dim

        # Allocate a "buffer" stack initialized with projected embeddings,
        # and maintain a cursor in this buffer.
        buffer_t = self._embedding_projection_network(
            raw_embeddings, self.word_embedding_dim, self.model_dim, self._vs, name="project")
        buffer_t = util.Dropout(buffer_t, self.embedding_dropout_keep_rate, self.apply_dropout)

        # Collapse buffer to (batch_size * buffer_size) * emb_dim for fast indexing.
        buffer_t = tf.reshape(buffer_t, (-1, self.model_dim))

        buffer_cur_t = tf.zeros((self.batch_size,), dtype=tf.int32, name="buffer_cur_init")

        # TODO(jgauthier): Implement linear memory (was in previous HardStack;
        # dropped it during a refactor)

        transitions = tf.transpose(self.transitions)
        stack, queue, cursors = self._stack, self._queue, self._cursors

        for t in range(self.seq_length):
            # Extract top buffer values.
            idxs = buffer_cur_t + (tf.range(0, self.batch_size) * self.seq_length)
            buffer_top_t = tf.gather(buffer_t, idxs)

            # if self._predict_network is not None:
            #     # We are predicting our own stack operations.
            #     predict_inp = T.concatenate(
            #         [stack_t[:, 0], stack_t[:, 1], buffer_top_t], axis=1)
            #     actions_t = self._predict_network(
            #         predict_inp, self.model_dim * 3, 2, self._vs,
            #         name="predict_actions")

            # if self.use_predictions:
            #     # Use predicted actions to build a mask.
            #     mask = actions_t.argmax(axis=1)
            # else:
            #     # Use transitions provided from external parser.
            #     mask = transitions_t
            mask = transitions[t, :]

            # stack1 values
            stack1_ptrs = tf.maximum(0, (t - 1) * self.batch_size + batch_range)
            stack1 = tf.gather(stack, stack1_ptrs, name="stack1_%i" % t)

            # stack2 values
            stack2_ptrs = tf.gather(
                    queue, tf.maximum(0, cursors - 1 + batch_range * self.seq_length))
            stack2 = tf.gather(stack, stack2_ptrs * self.batch_size + batch_range,
                               name="stack2_%i" % t)

            merge_items = tf.concat(1, [stack1, stack2])
            merge_value = self._compose_network(merge_items, self.model_dim * 2,
                                                self.model_dim, self._vs,
                                                name="compose")

            # Compute new stack value.
            stack_next, queue_next, cursors_next = update_hard_stack(
                    t, stack, buffer_top_t, merge_value, queue, cursors, mask,
                    self.batch_size, self.seq_length, batch_range)
            stack, queue, cursors = stack_next, queue_next, cursors_next

            # Move buffer cursor as necessary. Since mask == 1 when merge, we
            # should increment each buffer cursor by 1 - mask
            buffer_cur_t = buffer_cur_t + (1 - mask)

            # if self._predict_network is not None:
            #     return stack_next, actions_t, buffer_cur_next
            # else:
            #     return stack_next, buffer_cur_next

        self.final_stack = stack
        self.transitions_pred = None


class Model0(HardStack):

    def __init__(self, *args, **kwargs):
        kwargs["predict_network"] = None
        kwargs["use_predictions"] = False
        super(Model0, self).__init__(*args, **kwargs)


class Model1(HardStack):

    def __init__(self, *args, **kwargs):
        kwargs["predict_network"] = kwargs.get("predict_network", util.Linear)
        kwargs["use_predictions"] = False
        super(Model1, self).__init__(*args, **kwargs)


class Model2(HardStack):

    def __init__(self, *args, **kwargs):
        kwargs["predict_network"] = kwargs.get("predict_network", util.Linear)
        kwargs["use_predictions"] = True
        super(Model2, self).__init__(*args, **kwargs)
