import unittest

import numpy as np
import theano
from theano import tensor as T

from spinn.plain_rnn import RNN
from spinn.util import VariableStore, CropAndPad, IdentityLayer


class RNNTestCase(unittest.TestCase):

    """Basic functional tests for RNN with dummy data."""

    def _make_rnn(self, seq_length=4):
        self.embedding_dim = embedding_dim = 3
        self.vocab_size = vocab_size = 10
        self.seq_length = seq_length
        
        def compose_network(h_prev, inp, embedding_dim, model_dim, vs, name="compose"):
            # Just add the two embeddings!
            W = T.concatenate([T.eye(model_dim), T.eye(model_dim)], axis=0)
            i = T.concatenate([h_prev, inp], axis=1)
            return i.dot(W)

        X = T.imatrix("X")
        training_mode = T.scalar("training_mode")
        vs = VariableStore()
        embeddings = np.arange(vocab_size).reshape(
            (vocab_size, 1)).repeat(embedding_dim, axis=1)
        self.model = RNN(
            embedding_dim, embedding_dim, vocab_size, seq_length, compose_network,
            IdentityLayer, training_mode, None, vs,
            X=X, make_test_fn=True, initial_embeddings=embeddings)

    def test_basic_ff(self):
        self._make_rnn(4)

        X = np.array([
            [3, 1,  2, 0],
            [3, 2,  4, 5]
        ], dtype=np.int32)

        expected = np.array([[6, 6, 6],
                             [14, 14, 14]])

        ret = self.model.scan_fn(X, 1.0)
        np.testing.assert_almost_equal(ret, expected)


if __name__ == '__main__':
    unittest.main()
