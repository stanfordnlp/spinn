"""Theano-based sum-of-words implementations."""

import numpy as np
import theano

from theano import tensor as T
from spinn import util


class CBOW(object):
    """Plain sum of words encoder implementation.
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, _0, _1,
                 _2, _3, _4, vs, 
                 X=None,
                 initial_embeddings=None,
                 make_test_fn=False,
                 use_attention=False,
                 **kwargs):
        """Construct an RNN.

        Args:
            model_dim: Dimensionality of hidden state. Must equal word_embedding_dim.
            vocab_size: Number of unique tokens in vocabulary.
            compose_network: Blocks-like function which accepts arguments
              `prev_hidden_state, inp, inp_dim, hidden_dim, vs, name` (see e.g. `util.LSTMLayer`).
            training_mode: A Theano scalar indicating whether to act as a training model 
              with dropout (1.0) or to act as an eval model with rescaling (0.0).
            vs: VariableStore instance for parameter storage
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
            make_test_fn: If set, create a function to run a scan for testing.
            kwargs, _0, _1, _2, _3, _4: Ignored. meant to make the signature match the signature of HardStack().
        """

        assert model_dim == word_embedding_dim
        assert not use_attention or use_attention == "None"

        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim

        self.vocab_size = vocab_size

        self._vs = vs

        self.initial_embeddings = initial_embeddings

        self.X = X

        self._make_params()
        self._make_inputs()
        self._make_sum()

        if make_test_fn:
            assert False, "Not implemented."

    def _make_params(self):
        # Per-token embeddings.
        if self.initial_embeddings is not None:
            def EmbeddingInitializer(shape):
                return self.initial_embeddings
            self.embeddings = self._vs.add_param(
                    "embeddings", (self.vocab_size, self.word_embedding_dim), 
                    initializer=EmbeddingInitializer,
                    trainable=False,
                    savable=False)
        else:
            self.embeddings = self._vs.add_param(
                "embeddings", (self.vocab_size, self.word_embedding_dim))

    def _make_inputs(self):
        self.X = self.X or T.imatrix("X")

    def _make_sum(self):
        """Build the sequential composition / scan graph."""

        batch_size, seq_length = self.X.shape

        # Look up all of the embeddings that will be used.
        raw_embeddings = self.embeddings[self.X]  # batch_size * seq_length * emb_dim

        self.final_representations = T.sum(raw_embeddings, axis=1, keepdims=True, dtype="float32", acc_dtype="float32")
        self.transitions_pred = T.zeros((batch_size, 0))
        self.predict_transitions = False
        self.tracking_state_final = None
        
