"""Theano-based RNN implementations."""

import numpy as np
import theano

from theano import tensor as T
from spinn import util


class RNN(object):
    """Plain RNN encoder implementation. Can use any activation function.
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, _0, compose_network,
                 _1, training_mode, _2, vs, 
                 train_with_predicted_transitions=False, 
                 X=None,
                 initial_embeddings=None,
                 make_test_fn=False,
                 **kwargs):
        """Construct an RNN.

        Args:
            model_dim: Dimensionality of hidden state.
            vocab_size: Number of unique tokens in vocabulary.
            compose_network: Blocks-like function which accepts arguments
              `prev_hidden_state, inp, inp_dim, hidden_dim, vs, name` (see e.g. `util.LSTMLayer`).
            training_mode: A Theano scalar indicating whether to act as a training model 
              with dropout (1.0) or to act as an eval model with rescaling (0.0).
            vs: VariableStore instance for parameter storage
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
            make_test_fn: If set, create a function to run a scan for testing.
            kwargs, _0, _1, _2: Ignored. Meant to make the signature match the signature of HardStack().
        """

        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim
        self.vocab_size = vocab_size

        self._compose_network = compose_network

        self._vs = vs

        self.initial_embeddings = initial_embeddings

        self.training_mode = training_mode

        self.X = X
        
        self._make_params()
        self._make_inputs()
        self._make_scan()

        if make_test_fn:
            self.scan_fn = theano.function([self.X, self.training_mode],
                                           self.embeddings,
                                           on_unused_input='warn')

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

    def _step(self, inputs_cur_t, hidden_prev_t):
        hidden_state_cur_t = self._compose_network(hidden_prev_t, inputs_cur_t, 
            self.word_embedding_dim, self.model_dim, self._vs, name="rnn")

        return hidden_state_cur_t

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_size, seq_length = self.X.shape

        # Look up all of the embeddings that will be used.
        raw_embeddings = self.embeddings[self.X]  # batch_size * seq_length * emb_dim
        raw_embeddings = raw_embeddings.dimshuffle(1, 0, 2)

        # Initialize the hidden state.
        hidden_init = T.zeros((batch_size, self.model_dim))

        self.states = theano.scan(
                self._step,
                sequences=[raw_embeddings],
                outputs_info=[hidden_init])[0]

        self.final_representations = self.states[-1]
        self.transitions_pred = T.zeros((batch_size, 0))
        self.predict_transitions = False
        self.tracking_state_final = None
        
