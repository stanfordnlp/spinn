"""Theano-based stack implementations."""


import theano
from theano import tensor as T

from rembed import util


class HardStack(object):

    def __init__(self, embedding_dim, vocab_size, seq_length, vs,
                 unroll_scan=True, X=None):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self._vs = vs
        self.unroll_scan = unroll_scan

        self.X = X

        self._make_params()
        self._make_inputs()
        self._make_scan()

        self.scan_fn = theano.function([self.X], self.final_stack)

    def _make_params(self):
        # Per-token embeddings
        self.embeddings = self._vs.add_param(
            "embeddings", (self.vocab_size, self.embedding_dim))

        # Embedding composition map
        self.W = self._vs.add_param("composition_W", (self.embedding_dim * 2,
                                                      self.embedding_dim))

    def _make_inputs(self):
        self.X = self.X or T.matrix("X")

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_size = self.X.shape[0]
        max_stack_size = self.X.shape[1]

        # Stack batch is a 3D tensor.
        stack_shape = (batch_size, max_stack_size, self.embedding_dim)
        stack_init = T.zeros(stack_shape)

        # Allocate two helper stack copies (passed as non_seqs into scan.)
        stack_pushed = T.zeros(stack_shape)
        stack_merged = T.zeros(stack_shape)

        # TODO
        # Precompute embedding lookups.
        # embedding_lookups = self.embeddings[self.X]
        # print embedding_lookups.ndim

        def step(x_t, stack_t, stack_pushed, stack_merged):
            x_t_clean = T.maximum(0, x_t)
            embs_t = self.W[x_t_clean]

            # Build two copies of the stack batch: one where every stack
            # has received a push op, and one where every stack has
            # received a merge op.
            #
            # TODO is set_subtensor slow?
            #
            # Copy 1: Push.
            stack_pushed = T.set_subtensor(stack_pushed[:, 0], embs_t)
            stack_pushed = T.set_subtensor(stack_pushed[:, 1:], stack_t[:, 1:])

            # Copy 2: Merge.
            els_to_merge = stack_t[:, :2].reshape((-1, self.embedding_dim * 2))
            merged = els_to_merge.dot(self.W)
            stack_merged = T.set_subtensor(stack_merged[:, 0], merged)
            stack_merged = T.set_subtensor(stack_merged[:, 1:], stack_t[:, 2:])

            # Use special input flag -1 to switch between two stacks.
            stack_next = T.switch(T.eq(x_t, -1), stack_merged,
                                  stack_pushed)
            return stack_next

        if self.unroll_scan:
            stacks = util.unroll_scan(
                step, self.X, non_sequences=[stack_pushed, stack_merged],
                outputs_info=[stack_init], n_steps=self.seq_length)
        else:
            stacks = theano.scan(step, self.X,
                                 non_sequences=[stack_pushed, stack_merged],
                                 outputs_info=[stack_init])[0]

        self.final_stack = stacks[-1]


if __name__ == '__main__':
    embedding_dim = 10
    vocab_size = 5
    seq_length = 5

    X = T.imatrix("X")
    stack = HardStack(embedding_dim, vocab_size, seq_length,
                      util.VariableStore(), X=X)
