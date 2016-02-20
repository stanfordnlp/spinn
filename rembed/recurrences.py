"""
Implements the core recurrences for various stack models.

The recurrences described here are unrolled into bona-fide stacks
by the `rembed.stack` model.
"""

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

    def forward(self, inputs):
        raise NotImplementedError("abstract method")


class SharedRecurrenceMixin(object):
    """Mixin providing various shared components."""

    def __init__(self):
        raise RuntimeError("Don't instantiate me; I'm a mixin!")

    def _context_sensitive_shift(self, tracking_hidden, tracking_hidden_dim,
                                 buffer_top, layer=util.Linear):
        """
        Compute a buffer top representation by mixing buffer top and hidden state.
        """
        # Exclude the cell value from the computation.
        tracking_hidden = tracking_hidden[:, :hidden_dim]

        inp = T.concatenate([tracking_hidden, buffer_top], axis=1)
        inp_dim = self._spec.word_embedding_dim + tracking_hidden_dim
        return layer(inp, inp_dim, self._spec.model_dim, self._vs)

    def _tracking_lstm(self, inputs, network):
        c1, c2, buffer_top, tracking_hidden = inputs[:4]
        inp = T.concatenate([c1, c2, buffer_top], axis=1)
        return network(tracking_hidden, inp, self._spec.model_dim * 3,
                       self.tracking_lstm_hidden_dim, self._vs,
                       name="prediction_and_tracking")


class Model0(Recurrence, SharedRecurrenceMixin):

    def __init__(self, spec, vs, compose_network,
                 prediction_and_tracking_network=None,
                 use_tracking_lstm=False,
                 use_attention=False,
                 tracking_lstm_hidden_dim=8,
                 context_sensitive_shift=False,
                 context_sensitive_use_relu=False):
        super(Model0, self).__init__(spec, vs)
        self.extra_outputs = [1] if use_tracking_lstm else []
        self.predicts_transitions = False

        self._compose_network = compose_network
        self._prediction_and_tracking_network = prediction_and_tracking_network
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.use_attention = use_attention
        self.use_context_sensitive_shift = context_sensitive_shift
        self.context_sensitive_use_relu = context_sensitive_use_relu

    def forward(self, inputs):
        c1, c2, buffer_top = inputs[:3]
        if self.use_tracking_lstm:
            tracking_hidden = inputs[3]

        if self.use_context_sensitive_shift:
            assert self.use_tracking_lstm
            layer = util.ReLULayer if self.context_sensitive_use_relu else util.Linear
            buffer_top = self._context_sensitive_shift(
                tracking_hidden, self.tracking_lstm_hidden_dim, buffer_top,
                layer=layer)

        if self.use_tracking_lstm:
            tracking_hidden, _ = self._tracking_lstm(
                inputs, self._prediction_and_tracking_network)

        # Compute merge values.
        merge_items = T.concatenate([c1, c2], axis=1)
        if self.use_tracking_lstm:
            tracking_h_t = tracking_hidden[:, :self.tracking_lstm_hidden_dim]
            merge_value = self._compose_network(
                merge_items, tracking_h_t, self._spec.model_dim, self._vs,
                name="compose", external_state_dim=self.tracking_lstm_hidden_dim)
        else:
            merge_value = self._compose_network(
                merge_items, self._spec.model_dim * 2, self._spec.model_dim,
                self._vs, name="compose")

        if self.use_tracking_lstm:
            return [tracking_hidden], [merge_value, tracking_hidden]
        else:
            return [], [merge_value]
