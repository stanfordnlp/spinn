import unittest

import numpy as np
import theano
from theano import tensor as T

from rembed.stack import HardStack
from rembed.util import VariableStore


def test_hard_stack():
    """Basic functional test for HardStack with dummy data."""

    embedding_dim = 3
    vocab_size = 6
    seq_length = 4
    num_composition_layers = 1

    def compose_network(inp, inp_dim, outp_dim, vs, name="compose"):
        # Just add the two embeddings!
        W = T.concatenate([T.eye(outp_dim), T.eye(outp_dim)], axis=0)
        return inp.dot(W)

    X = T.imatrix("X")
    transitions = T.imatrix("transitions")
    vs = VariableStore()
    stack = HardStack(
        embedding_dim, vocab_size, seq_length, compose_network,
        vs,
        X=X, transitions=transitions)

    # Swap in our own dummy embeddings and weights.
    embeddings = np.arange(vocab_size).reshape((vocab_size, 1)).repeat(embedding_dim, axis=1)
    stack.embeddings.set_value(embeddings)

    X = np.array([
        [3, 1,  2, 0],
        [3, 2,  4, 5]
    ], dtype=np.int32)

    transitions = np.array([
        # First input: push a bunch onto the stack
        [0, 0, 0, 0],
        # Second input: push, then merge, then push more. (Leaves one item on
        # the buffer.)
        [0, 0, 1, 0]
    ], dtype=np.int32)

    expected = np.array([[[0, 0, 0],
                          [2, 2, 2],
                          [1, 1, 1],
                          [3, 3, 3]],
                         [[4, 4, 4],
                          [5, 5, 5],
                          [0, 0, 0],
                          [0, 0, 0]]])

    ret = stack.scan_fn(X, transitions)
    np.testing.assert_almost_equal(ret, expected)

if __name__ == '__main__':
    test_hard_stack()
