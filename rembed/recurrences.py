"""
Implements the core recurrences for various stack models.

The recurrences described here are unrolled into bona-fide stacks
by the `rembed.stack` model.
"""

from functools import partial

from rembed import util


class Recurrence(object):

    def __init__(self, spec, vs):
        self._spec = spec
        self._vs = vs

        # A recurrence is expected to output 1 value in a merge op and zero
        # values in a push op. A recurrence may also output `N` "extra"
        # outputs, expected at both merge and push ops.
        #
        # This list should contain `N` integers. The integer at position `i`
        # specifies that extra output #`i` will have `extra_outputs[i]`
        # dimensions given a single example (non-batched) input.
        self.extra_outputs = []

        self.predicts_transitions = False
        self.uses_transitions = False

    def forward(self, inputs, **constants):
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
        c1, c2, buffer_top, tracking_hidden = inputs[:4]
        inp = T.concatenate([c1, c2, buffer_top], axis=1)
        return network(tracking_hidden, inp, self._spec.model_dim * 3,
                       self.tracking_lstm_hidden_dim, self._vs,
                       name="prediction_and_tracking")

    def _predict(self, inputs, network):
        c1, c2, buffer_top = inputs[:3]
        inp = T.concatenate([c1, c2, buffer_top], axis=1)
        return network(inp, self._spec.model_dim * 3,
                       util.NUM_TRANSITION_TYPES, self._vs,
                       name="prediction_and_tracking")

    def _merge(self, inputs, network):
        c1, c2 = inputs[:2]
        merge_items = T.concatenate([c1, c2], axis=1)
        if self.use_tracking_lstm:
            tracking_h_t = inputs[2][:, :self.tracking_lstm_hidden_dim]
            return network(merge_items, tracking_h_t, self._spec.model_dim,
                           self._vs, name="compose",
                           external_state_dim=self.tracking_lstm_hidden_dim)
        else:
            return network(merge_items, self._spec.model_dim * 2,
                           self._spec.model_dim, self._vs, name="compose")


class Model0(Recurrence, SharedRecurrenceMixin):

    def __init__(self, spec, vs, compose_network,
                 use_tracking_lstm=False,
                 tracking_lstm_hidden_dim=8,
                 context_sensitive_shift=False,
                 context_sensitive_use_relu=False):
        super(Model0, self).__init__(spec, vs)
        self.extra_outputs = []
        if use_tracking_lstm:
            self.extra_outputs.append(1)
        self.predicts_transitions = False

        self._compose_network = compose_network
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.use_context_sensitive_shift = context_sensitive_shift
        self.context_sensitive_use_relu = context_sensitive_use_relu

        if use_tracking_lstm:
            self._prediction_and_tracking_network = partial(util.TrackingUnit,
                                                            make_logits=False)

    def forward(self, inputs, **constants):
        c1, c2, buffer_top = inputs[:3]
        if self.use_tracking_lstm:
            tracking_hidden = inputs[3]

        if self.use_context_sensitive_shift:
            buffer_top = self._context_sensitive_shift(inputs)

        if self.use_tracking_lstm:
            tracking_hidden, _ = self._tracking_lstm_predict(
                inputs, self._prediction_and_tracking_network)

        merge_value = self._merge(inputs, network)

        if self.use_tracking_lstm:
            return [tracking_hidden], [merge_value, tracking_hidden]
        else:
            return [], [merge_value]


class Model1(Recurrence, SharedRecurrenceMixin):

    def __init__(self, spec, vs, compose_network,
                 use_tracking_lstm=False,
                 tracking_lstm_hidden_dim=8,
                 context_sensitive_shift=False,
                 context_sensitive_use_relu=False):
        super(Model1, self).__init__(spec, vs)
        self.extra_outputs = [1]
        if use_tracking_lstm:
            self.extra_outputs.append(1)
        self.predicts_transitions = True
        self.uses_transitions = False

        self._compose_network = compose_network
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.use_context_sensitive_shift = context_sensitive_shift
        self.context_sensitive_use_relu = context_sensitive_use_relu

        if use_tracking_lstm:
            self._prediction_and_tracking_network = partial(util.TrackingUnit,
                                                            make_logits=False)
        else:
            self._prediction_and_tracking_network = util.Linear

    def forward(self, inputs, **constants):
        c1, c2, buffer_top = inputs[:3]
        if self.use_tracking_lstm:
            tracking_hidden = inputs[3]

        if self.use_context_sensitive_shift:
            buffer_top = self._context_sensitive_shift(inputs)

        # Predict transitions.
        if self.use_tracking_lstm:
            tracking_hidden, actions_t = self._tracking_lstm_predict(
                inputs, self._prediction_and_tracking_network)
        else:
            actions_t = self._predict(
                inputs, self._prediction_and_tracking_network)

        merge_value = self._merge(inputs, network)

        if self.use_tracking_lstm:
            return [tracking_hidden], [merge_value, tracking_hidden], actions_t
        else:
            return [], [merge_value], actions_t


class Model2(Model1, SharedRecurrenceMixin):

    def __init__(self, spec, vs, compose_network, **kwargs):
        super(Model2, self).__init__(spec, vs, compose_network, **kwargs)
        self.uses_transitions = True
