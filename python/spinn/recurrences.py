"""
Implements the core recurrences for various stack models.

The recurrences described here are unrolled into bona-fide stack models
by `spinn.stack`.
"""

from functools import partial

from theano import tensor as T

from spinn import util


class Recurrence(object):

    def __init__(self, spec, vs):
        self._spec = spec
        self._vs = vs

        # TODO(SB): Standardize terminology in comments --
        #   Merge/push v. push/pop v. shift/reduce...

        # A recurrence is expected to output 1 value in a merge op and zero
        # values in a push op. A recurrence may also output `N` "extra"
        # outputs, expected at both merge and push ops.
        #
        # This list should contain `N` shape tuples. The tuple at position `i`
        # specifies that extra output #`i` will have shape `extra_outputs[i]`
        # for a single example (i.e., not including batch axis). For example,
        # if a recurrence yields a 50-dimensional vector for each example at
        # each timestep, we would include `(50,)` here.
        self.extra_outputs = []

        self.predicts_transitions = False
        self.uses_predictions = False

    def __call__(self, inputs, **constants):
        """
        Computes push and merge results for a single timestep.

        Args:
            inputs: A tuple of inputs to the recurrence. At minimum this
                contains three elements: `(stack_1, stack_2, buffer_top)`.
                A recurrence which has `N` extra outputs will also receive
                all `N` outputs from the previous timestep concatenated to
                this tuple, e.g.

                    (stack_1, stack_2, buffer_top,
                     prev_output_1, prev_output_2, ...)

                Each input is a batch of values, with batch_size as the leading
                axis.
            constants:
                TBD

        Returns:
            push_outputs: A list of batch outputs for the case in which the
                current op is a push. This list should be `self.extra_outputs`
                long.
            merge_outputs: A list of batch outputs for the case in which the
                current op is a merge. This list should be
                `1 + self.extra_outputs` long. (The first element should be
                the result of merging the stack top values, a batch tensor of
                shape `batch_size * self._spec.model_dim`.)
            actions: (Only necessary if `self.predicts_transitions` is `True`.)
                A batch of logits over stack actions at this timestep (of shape
                `batch_size * num_actions`).
        """
        raise NotImplementedError("abstract method")


class SharedRecurrenceMixin(object):
    """Mixin providing various shared components."""

    def __init__(self):
        raise RuntimeError("Don't instantiate me; I'm a mixin!")

    def _context_sensitive_shift(self, inputs):
        """
        Compute a buffer top representation by mixing buffer top and hidden state.

        NB: This hasn't been an especially effective tool so far.
        """
        assert self.use_tracking_lstm
        buffer_top, tracking_hidden = inputs[2:4]

        # Exclude the cell value from the computation.
        tracking_hidden = tracking_hidden[:, :hidden_dim]

        inp = T.concatenate([tracking_hidden, buffer_top], axis=1)
        inp_dim = self._spec.word_embedding_dim + self.tracking_lstm_hidden_dim
        layer = util.ReLULayer if self.context_sensitive_use_relu else util.Linear
        return layer(inp, inp_dim, self._spec.model_dim, self._vs,
                     name="context_comb_unit", use_bias=True,
                     initializer=util.HeKaimingInitializer())

    def _tracking_lstm_predict(self, inputs, network):
        # TODO(SB): Offer more buffer content than just the top as input.
        c1, c2, buffer_top, tracking_hidden = inputs[:4]

        h_dim = self._spec.model_dim
        if self._spec.model_visible_dim != self._spec.model_dim:
            h_dim = self._spec.model_visible_dim
            c1 = c1[:, :h_dim]
            c2 = c2[:, :h_dim]
            buffer_top = buffer_top[:, :h_dim]

        inp = (c1, c2, buffer_top)
        return network(tracking_hidden, inp, (h_dim,) * 3,
                       self.tracking_lstm_hidden_dim, self._vs,
                       name="prediction_and_tracking")

    def _predict(self, inputs, network):
        # TODO(SB): Offer more buffer content than just the top as input.
        c1, c2, buffer_top = tuple(inputs[:3])

        h_dim = self._spec.model_dim
        if self._spec.model_visible_dim != self._spec.model_dim:
            h_dim = self._spec.model_visible_dim
            c1 = c1[:, :h_dim]
            c2 = c2[:, :h_dim]
            buffer_top = buffer_top[:, :h_dim]

        inp = (c1, c2, buffer_top)
        return network(inp, (h_dim,) * 3, util.NUM_TRANSITION_TYPES, self._vs,
                       name="prediction_and_tracking")

    def _merge(self, inputs, network):
        merge_items = tuple(inputs[:2])
        if self.use_tracking_lstm:
            # NB: Unlike in the previous implementation, context-sensitive
            # composition (aka the tracking--composition connection) is not
            # optional here. It helps performance, so this shouldn't be a
            # big problem.
            tracking_h_t = inputs[3][:, :self.tracking_lstm_hidden_dim]
            return network(merge_items, tracking_h_t, self._spec.model_dim,
                           self._vs, name="compose",
                           external_state_dim=self.tracking_lstm_hidden_dim)
        else:
            return network(merge_items, (self._spec.model_dim,) * 2,
                           self._spec.model_dim, self._vs, name="compose")


class Model0(Recurrence, SharedRecurrenceMixin):

    def __init__(self, spec, vs, compose_network,
                 use_tracking_lstm=False,
                 tracking_lstm_hidden_dim=8,
                 use_context_sensitive_shift=False,
                 context_sensitive_use_relu=False):
        super(Model0, self).__init__(spec, vs)
        self.extra_outputs = []
        if use_tracking_lstm:
            self.extra_outputs.append((tracking_lstm_hidden_dim * 2,))
        self.predicts_transitions = False

        self._compose_network = compose_network
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.use_context_sensitive_shift = use_context_sensitive_shift
        self.context_sensitive_use_relu = context_sensitive_use_relu

        if use_tracking_lstm:
            self._prediction_and_tracking_network = partial(util.TrackingUnit,
                                                            make_logits=False)

    def __call__(self, inputs, **constants):
        c1, c2, buffer_top = inputs[:3]
        if self.use_tracking_lstm:
            tracking_hidden = inputs[3]

        # Unlike in the previous implementation, we update the tracking LSTM
        # before using its output to update the inputs.
        if self.use_tracking_lstm:
            tracking_hidden, _ = self._tracking_lstm_predict(
                inputs, self._prediction_and_tracking_network)
            inputs = [c1, c2, buffer_top, tracking_hidden]

        if self.use_context_sensitive_shift:
            buffer_top = self._context_sensitive_shift(inputs)

        merge_value = self._merge(inputs, self._compose_network)

        if self.use_tracking_lstm:
            return [tracking_hidden], [merge_value, tracking_hidden]
        else:
            return [], [merge_value]


class Model1(Recurrence, SharedRecurrenceMixin):

    def __init__(self, spec, vs, compose_network,
                 use_tracking_lstm=False,
                 tracking_lstm_hidden_dim=8,
                 use_context_sensitive_shift=False,
                 context_sensitive_use_relu=False):
        super(Model1, self).__init__(spec, vs)
        if use_tracking_lstm:
            self.extra_outputs.append((tracking_lstm_hidden_dim * 2,))
        self.predicts_transitions = True
        self.uses_predictions = False

        self._compose_network = compose_network
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.use_context_sensitive_shift = use_context_sensitive_shift
        self.context_sensitive_use_relu = context_sensitive_use_relu

        if use_tracking_lstm:
            self._prediction_and_tracking_network = partial(util.TrackingUnit,
                                                            make_logits=False)
        else:
            self._prediction_and_tracking_network = util.Linear

    def __call__(self, inputs, **constants):
        c1, c2, buffer_top = inputs[:3]
        if self.use_tracking_lstm:
            tracking_hidden = inputs[3]

        # Predict transitions.

        # Unlike in the previous implementation, we update the tracking LSTM
        # before using its output to update the inputs.
        if self.use_tracking_lstm:
            tracking_hidden, actions_t = self._tracking_lstm_predict(
                inputs, self._prediction_and_tracking_network)
            inputs = [c1, c2, buffer_top, tracking_hidden]
        else:
            actions_t = self._predict(
                inputs, self._prediction_and_tracking_network)

        if self.use_context_sensitive_shift:
            buffer_top = self._context_sensitive_shift(inputs)

        merge_value = self._merge(inputs, self._compose_network)

        if self.use_tracking_lstm:
            return [tracking_hidden], [merge_value, tracking_hidden], actions_t
        else:
            return [], [merge_value], actions_t


class Model2(Model1, SharedRecurrenceMixin):
    """Core implementation of Model 2. Supports scheduled sampling."""

    def __init__(self, spec, vs, compose_network, **kwargs):
        super(Model2, self).__init__(spec, vs, compose_network, **kwargs)
        self.uses_predictions = True
