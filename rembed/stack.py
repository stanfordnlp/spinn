"""Theano-based stack implementations."""


import theano
from theano import tensor as T

from rembed import util


def update_hard_stack(stack_t, stack_pushed, stack_merged, push_value,
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
    stack_next = mask * stack_merged + (1 - mask) * stack_pushed

    return stack_next


class HardStack(object):

    def __init__(
        self, embedding_dim, vocab_size, seq_length, compose_network, vs,
        linear_memory_dim=None, X=None):
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
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
        """

        self.embedding_dim = embedding_dim
        self.linear_memory_dim = linear_memory_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self._compose_network = compose_network
        self._vs = vs

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
        use_linear_memory = self.linear_memory_dim is not None
        linear_memory_init = T.zeros(
            (batch_size, (self.linear_memory_dim or 0) * 2))

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
            if use_linear_memory:
                next_linear_memory = util.LSTM(
                    linear_memory, embs_t, self.embedding_dim,
                    self.linear_memory_dim, self._vs, name="linear_memory")
                # Only update memories for examples which are not merging
                mask_mem = mask.dimshuffle(0, "x")
                linear_memory = (mask_mem * linear_memory
                                 + (1 - mask_mem) * next_linear_memory)

            # Precompute all merge values.
            merge_items = stack_t[:, :2].reshape((-1, self.embedding_dim * 2))
            merge_value = self._compose_network(
                merge_items, self.embedding_dim * 2, self.embedding_dim,
                self._vs, name="compose")

            # Update helper stack values.
            stack_next = update_hard_stack(
                stack_t, stack_pushed, stack_merged, embs_t, merge_value, mask)

            return stack_next, linear_memory

        # Dimshuffle inputs to seq_len * batch_size for scanning
        X = self.X.dimshuffle(1, 0)

        scan_ret = theano.scan(
            step, X, non_sequences=[stack_pushed, stack_merged],
            outputs_info=[stack_init, linear_memory_init])[0]

        if use_linear_memory:
            self.final_stack = scan_ret[0][-1]
            self.final_linear = scan_ret[1][-1][:, :self.linear_memory_dim]
        else:
            self.final_stack = scan_ret[0][-1]
            self.final_linear = None


class Model1(object):

    """Model 1 stack implementation.

    Model 1 scans a sequence using a hard stack. It predicts stack operations
    using an MLP, and receives supervision on these predictions from some
    external parser which acts as the "ground truth" parser.
    """

    def __init__(self, embedding_dim, vocab_size, seq_length, compose_network,
                 vs, predict_network=None, X=None, transitions=None):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self._compose_network = compose_network
        self._predict_network = predict_network

        self._vs = vs

        self.X = X
        self.transitions = transitions

        self._make_params()
        self._make_inputs()
        self._make_scan()

        self.scan_fn = theano.function([self.X], self.final_stack)

    def _make_params(self):
        # Per-token embeddings.
        self.embeddings = self._vs.add_param(
            "embeddings", (self.vocab_size, self.embedding_dim))

    def _make_inputs(self):
        self.X = self.X or T.matrix("X")
        self.transitions = self.transitions or T.matrix("transitions")

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_size, max_stack_size = self.X.shape

        # Stack batch is a 3D tensor.
        stack_shape = (batch_size, max_stack_size, self.embedding_dim)
        stack_init = T.zeros(stack_shape)

        # Allocate two helper stack copies (passed as non_seqs into scan).
        stack_pushed = T.zeros(stack_shape)
        stack_merged = T.zeros(stack_shape)

        # Allocate a "buffer" stack and maintain a cursor in this buffer.
        # TODO(jgauthier): Verify shape.
        buffer = self.embeddings[self.X]
        assert buffer.ndim == 3
        buffer_cur_init = T.zeros((batch_size,))

        def step(x_t, stack_t, buffer_cur_t, stack_pushed, stack_merged,
                 buffer):
            embs_t = self.embeddings[x_t]

            # Extract top buffer values.
            buffer_top_t = buffer[T.arange(batch_size), buffer_cur_t]

            # Predict stack actions.
            predict_inp = T.concatenate(
                [x_t, stack_t[:, 0], buffer_top_t], axis=1)
            actions_t = self._predict_network(
                predict_inp, self.embedding_dim * 3, 2, self._vs,
                name="predict_actions")

            # Use predicted actions to build a mask.
            mask = actions_t.argmax(axis=1)
            assert mask.ndim == 1

            # Now update the stack: first precompute merge results.
            merge_items = stack_t[:, :2].reshape((-1, self.embedding_dim * 2))
            merge_value = self._compose_network(
                merge_items, self.embedding_dim * 2, self.embedding_dim,
                self._vs, name="compose")

            # Compute new stack value.
            stack_next = update_hard_stack(
                stack_t, stack_pushed, stack_merged, embs_t, merge_value, mask)

            # Move buffer cursor as necessary. Since mask == 1 when merge, we
            # should increment each buffer cursor by 1 - mask
            buffer_cur_next = buffer_cur_t + (1 - mask)

            return stack_next, buffer_cur_next

        # Dimshuffle inputs to seq_len * batch_size for scanning
        X = self.X.dimshuffle(1, 0)

        scan_ret = theano.scan(
            step, X, non_sequences=[stack_pushed, stack_merged, buffer],
            outputs_info=[stack_init, buffer_cur_init])[0]

        self.final_stack = scan_ret[0][-1]


if __name__ == '__main__':
    embedding_dim = 10
    vocab_size = 5
    seq_length = 5

    X = T.imatrix("X")
    stack = HardStack(embedding_dim, vocab_size, seq_length,
                      util.VariableStore(), X=X)
