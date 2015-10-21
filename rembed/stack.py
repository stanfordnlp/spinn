"""Theano-based stack implementations."""


import theano
from theano import tensor as T

from rembed import util


class HardStack(object):

    def __init__(
        self, embedding_dim, vocab_size, seq_length, compose_network, vs,
        linear_memory_dim=None, unroll_scan=True, X=None):
        """Construct a HardStack.

        Args:
            embedding_dim: Dimensionality of token embeddings and stack values
            vocab_size: Number of unique tokens in vocabulary
            seq_length: Maximum sequence length which will be processed by this
              stack
            compose_network: Blocks-like function which accepts arguments
              `inp, inp_dim, outp_dim, vs, name` (see e.g. `util.Linear`).
              Given a Theano batch `inp` of dimension `batch_size * inp_dim`,
              returns a transformed Theano batch of dimension
              `batch_size * outp_dim`.
            vs: VariableStore instance for parameter storage
            linear_memory_dim: Integer or `None`. If not `None`, maintain an
              LSTM memory while scanning input sequences (a "linear" memory as
              opposed to the main stack-based memory). When linear memory
              is active, this instance will have a batch `final_memory`
              of dimension `batch_size * linear_memory_dim`.
            unroll_scan: If `True`, expand the recurrent scan over the input
              sequence without using `theano.scan`. This makes for faster
              compilation, but may preclude nice optimizations. It also will
              break when `seq_length` is high (~ >80), as it yields an
              excessively large computation graph.
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
        """

        self.embedding_dim = embedding_dim
        self.linear_memory_dim = linear_memory_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self._compose_network = compose_network
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

        # Allocate linear memory.
        if self.linear_memory_dim is not None:
            linear_memory = T.zeros((batch_size, self.linear_memory_dim * 2))
        else:
            linear_memory = None

        # TODO
        # Precompute embedding lookups.
        # embedding_lookups = self.embeddings[self.X]
        # print embedding_lookups.ndim

        def step(x_t, stack_t, linear_memory, stack_pushed, stack_merged):
            # NB: x_t may contain sentinel -1 values. Luckily -1 is a
            # valid index, and the below lookup doesn't fail. In any
            # case, where x_t we won't use the resultant embedding
            # anyway!
            embs_t = self.embeddings[x_t]

            # Mask to select examples which have a "merge" op at this
            # timestep
            mask = T.eq(x_t, -1)

            # If we are carrying along a linear memory, update that first.
            if linear_memory is not None:
                next_linear_memory = util.LSTM(
                    linear_memory, embs_t, self.embedding_dim,
                    self.linear_memory_dim, self._vs, name="linear_memory")
                # Only update memories for examples which are not merging
                mask_mem = mask.dimshuffle(0, "x")
                linear_memory = (mask_mem * linear_memory
                                 + (1 - mask_mem) * next_linear_memory)

            # Build two copies of the stack batch: one where every stack
            # has received a push op, and one where every stack has
            # received a merge op.
            #
            # TODO is set_subtensor slow?
            #
            # Copy 1: Push.
            stack_pushed = T.set_subtensor(stack_pushed[:, 0], embs_t)
            stack_pushed = T.set_subtensor(
                stack_pushed[:, 1:], stack_t[:, :-1])

            # Copy 2: Merge.
            merge_items = stack_t[:, :2].reshape((-1, self.embedding_dim * 2))
            merged = self._compose_network(merge_items, self.embedding_dim * 2,
                                           self.embedding_dim, self._vs,
                                           name="compose")

            stack_merged = T.set_subtensor(stack_merged[:, 0], merged)
            stack_merged = T.set_subtensor(
                stack_merged[:, 1:-1], stack_t[:, 2:])

            # Use special input flag -1 to switch between two stacks.
            mask_stack = mask.dimshuffle(0, "x", "x")
            stack_next = (mask_stack * stack_merged
                          + (1 - mask_stack) * stack_pushed)

            return stack_next, linear_memory

        # Dimshuffle inputs to seq_len * batch_size for scanning
        X = self.X.dimshuffle(1, 0)

        if self.unroll_scan:
            scan_ret = util.unroll_scan(
                step, X, non_sequences=[stack_pushed, stack_merged],
                outputs_info=[stack_init, linear_memory],
                n_steps=self.seq_length)[0]
        else:
            scan_ret = theano.scan(
                step, X, non_sequences=[stack_pushed, stack_merged],
                outputs_info=[stack_init, linear_memory])[0]

        if linear_memory is not None:
            self.final_stack = scan_ret[0][-1]
            self.final_linear = scan_ret[1][-1][:, :self.linear_memory_dim]
        else:
            self.final_stack = scan_ret[-1]
            self.final_linear = None


if __name__ == '__main__':
    embedding_dim = 10
    vocab_size = 5
    seq_length = 5

    X = T.imatrix("X")
    stack = HardStack(embedding_dim, vocab_size, seq_length,
                      util.VariableStore(), X=X)
