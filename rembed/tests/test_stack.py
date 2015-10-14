import unittest

import numpy as np
import theano
from theano import tensor as T

from rembed.stack import HardStack
from rembed.util import VariableStore


def test_hard_stack():
    """Basic functional test for HardStack with dummy data."""

    embedding_dim = 3
    vocab_size = 5
    seq_length = 4

    X = T.imatrix("X")
    stack = HardStack(embedding_dim, vocab_size, seq_length, VariableStore(),
                      X=X)

    # Swap in our own dummy embeddings and weights.
    embeddings = np.arange(5).reshape((5, 1)).repeat(3, axis=1)
    W = np.vstack([np.eye(embedding_dim), np.eye(embedding_dim)])

    stack.embeddings.set_value(embeddings)
    stack.W.set_value(W)

    X = np.array([
        # First input: push a bunch onto the stack
        [3, 1,  2, 0],
        # Second input: push, then merge, then push more
        [3, 2, -1, 4]
        ], dtype=np.int32)

    expected = np.array([[[0, 0, 0],
                          [2, 2, 2],
                          [1, 1, 1],
                          [3, 3, 3]],
                         [[4, 4, 4],
                          [5, 5, 5],
                          [0, 0, 0],
                          [0, 0, 0]]])

    ret = stack.scan_fn(X)

    np.testing.assert_almost_equal(ret, expected)
