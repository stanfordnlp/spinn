import unittest

import numpy as np
import theano
from theano import tensor as T

from rembed.stack import HardStack
from rembed.util import VariableStore, CropAndPad, IdentityLayer, batch_subgraph_gradients


class HardStackTestCase(unittest.TestCase):

    """Basic functional tests for HardStack with dummy data."""

    def _make_stack(self, seq_length=4):
        self.batch_size = 2
        self.embedding_dim = embedding_dim = 3
        self.vocab_size = vocab_size = 10
        self.seq_length = seq_length

        def compose_network(inp, inp_dim, outp_dim, vs, name="compose"):
            # Just add the two embeddings!
            W = T.concatenate([T.eye(outp_dim), T.eye(outp_dim)], axis=0)
            return inp.dot(W)

        X = T.imatrix("X")
        transitions = T.imatrix("transitions")
        training_mode = T.scalar("training_mode")
        ground_truth_transitions_visible = T.scalar("ground_truth_transitions_visible", dtype="int32")
        vs = VariableStore()

        # Swap in our own dummy embeddings and weights.
        initial_embeddings = np.arange(vocab_size).reshape(
            (vocab_size, 1)).repeat(embedding_dim, axis=1)

        self.stack = HardStack(
            embedding_dim, embedding_dim, 2, vocab_size, seq_length, compose_network,
            IdentityLayer, training_mode, ground_truth_transitions_visible, vs,
            X=X,
            transitions=transitions,
            make_test_fn=True,
            initial_embeddings=initial_embeddings,
            use_input_batch_norm=False, use_input_dropout=False)

    def test_basic_ff(self):
        self._make_stack(4)

        X = np.array([
            [3, 1,  2, 0],
            [3, 2,  4, 5]
        ], dtype=np.int32)

        transitions = np.array([
            # First input: push a bunch onto the stack
            [0, 0, 0, 0],
            # Second input: push, then merge, then push more. (Leaves one item
            # on the buffer.)
            [0, 0, 1, 0]
        ], dtype=np.int32)

        expected = np.array([[ 3.,  3.,  3.],
                             [ 3.,  3.,  3.],
                             [ 1.,  1.,  1.],
                             [ 2.,  2.,  2.],
                             [ 2.,  2.,  2.],
                             [ 5.,  5.,  5.],
                             [ 0.,  0.,  0.],
                             [ 4.,  4.,  4.]])

        self.stack.scan_fn(X, transitions, 1.0, 1)
        ret = self.stack.stack.get_value()
        np.testing.assert_almost_equal(ret, expected)

    def test_with_cropped_data(self):
        dataset = [
            {
                "tokens": [1, 2, 4, 5, 6, 2],
                "transitions": [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
            },
            {
                "tokens": [6, 1],
                "transitions": [0, 0, 1]
            },
            {
                "tokens": [6, 1, 2, 3, 5, 1],
                "transitions": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            }
        ]

        seq_length = 5
        self._make_stack(seq_length)

        dataset = CropAndPad(dataset, seq_length)
        X = np.array([example["tokens"] for example in dataset],
                     dtype=np.int32)
        transitions = np.array([example["transitions"] for example in dataset],
                               dtype=np.int32)
        expected = np.array([[[8, 8, 8],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[7, 7, 7],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]]])

        ret = self.stack.scan_fn(X, transitions, 1.0, 1)
        np.testing.assert_almost_equal(ret, expected)


class ThinStackBackpropTestCase(unittest.TestCase):

    def setUp(self):
        if 'gpu' not in theano.config.device:
            raise RuntimeError("Thin stack only defined for GPU usage")

        self.embedding_dim = self.model_dim = 2
        self.vocab_size = 5
        self.seq_length = 5
        self.batch_size = 2
        self.num_classes = 2

        embeddings = (np.arange(self.vocab_size)[:, np.newaxis]
                .repeat(self.embedding_dim, axis=1).astype(np.float32) + 1.0)
        W = theano.shared(np.array([[ 0.09853827,  0.28727029],
                                    [ 0.70784546,  0.17831399],
                                    [ 0.96303163,  0.53989795],
                                    [ 0.37782846,  0.83950132]],
                                   dtype=np.float32))
        b = theano.shared(np.array([1.0, 1.0], dtype=np.float32))
        self.embeddings = embeddings
        self.W = W
        self.b = b

        self.compose_network = lambda inp, *args, **kwargs: T.dot(inp, W) + b

        def ghost_compose_net(c1, c2):
            if c1.ndim == 1: c1 = c1[np.newaxis, :]
            if c2.ndim == 1: c2 = c2[np.newaxis, :]
            return self.compose_network(T.concatenate([c1, c2], axis=1)).squeeze()
        self.ghost_compose_net = ghost_compose_net

        self.X = T.imatrix("X")
        self.transitions = T.imatrix("transitions")
        self.y = T.ivector("y")

        self.stack = HardStack(
            self.model_dim, self.embedding_dim, self.batch_size,
            self.vocab_size, self.seq_length, self.compose_network,
            IdentityLayer, 0.0, 1.0, VariableStore(),
            X=self.X,
            transitions=self.transitions,
            initial_embeddings=embeddings,
            use_input_batch_norm=False,
            use_input_dropout=False)

    def _fake_stack_ff(self):
        """Fake a stack feedforward S S M S M with the given data."""

        # seq_length * batch_size * emb_dim
        X_emb = self.stack.embeddings[self.X].dimshuffle(1, 0, 2)

        z1 = T.concatenate([X_emb[1], X_emb[0]], axis=1)
        c1 = self.compose_network(z1)

        z2 = T.concatenate([X_emb[2], c1], axis=1)
        c2 = self.compose_network(z2)

        return c2

    def _make_cost(self, stack_top):
        logits = stack_top[:, :self.num_classes]
        costs = T.nnet.categorical_crossentropy(T.nnet.softmax(logits), self.y)
        return costs.mean()

    def test_backprop(self):
        # Simulate a batch of two token sequences, each with the same
        # transition sequence
        X = np.array([[0, 1, 2, 3, 1], [2, 1, 3, 0, 1]], dtype=np.int32)
        y = np.array([1, 0], dtype=np.int32)
        transitions = np.tile([0, 0, 1, 0, 1], (2, 1)).astype(np.int32)

        simulated_top = self._fake_stack_ff()
        simulated_cost = self._make_cost(simulated_top)
        f_simulated = theano.function(
            [self.X, self.y],
            (simulated_cost, T.grad(simulated_cost, self.W),
             T.grad(simulated_cost, self.b)))

        top = self.stack.final_stack[-self.batch_size:]
        cost = self._make_cost(top)
        error_signal = T.grad(cost, top)

        # Build composition gradient subgraph.
        f_delta1 = batch_subgraph_gradients([1, 1], [self.W, self.b], self.ghost_compose_net)
        f_delta = lambda (c1, c2, buffer_top), (grad, _): f_delta1([c1, c2], [grad])

        # Now build backprop, passing in our composition gradient.
        self.stack.make_backprop_scan([], error_signal, f_delta,
                                      [self.W.get_value().shape,
                                       self.b.get_value().shape])
        f = theano.function(
            [self.X, self.transitions, self.y],
            (cost, self.stack.deltas[0], self.stack.deltas[1]))

        b_cost_sim, b_dW_sim, b_db_sim = f_simulated(X, y)
        b_cost, b_dW, b_db = f(X, transitions, y)

        np.testing.assert_almost_equal(b_cost_sim, b_cost)
        np.testing.assert_almost_equal(b_dW_sim, b_dW)
        np.testing.assert_almost_equal(b_db_sim, b_db)


class ThinStackTrackingBackpropTestCase(unittest.TestCase):

    def setUp(self):
        if 'gpu' not in theano.config.device:
            raise RuntimeError("Thin stack only defined for GPU usage")

        self.embedding_dim = self.model_dim = 2
        self.vocab_size = 5
        self.seq_length = 5
        self.batch_size = 2
        self.num_classes = 2

        embeddings = (np.arange(self.vocab_size)[:, np.newaxis]
                .repeat(self.embedding_dim, axis=1).astype(np.float32) + 1.0)
        W = theano.shared(np.array([[ 0.81020794,  0.39202386],
                                    [ 0.04252092,  0.12962219],
                                    [ 0.78306797,  0.81535813],
                                    [ 0.86703663,  0.04581873],
                                    [ 0.80502693,  0.08334766]],
                                   dtype=np.float32))
        b = theano.shared(np.array([1.0, 1.0], dtype=np.float32))
        W_track = theano.shared(np.array([[ 0.77254916,  0.42812964],
                                          [ 0.88963778,  0.44141495],
                                          [ 0.04797265,  0.26553046],
                                          [ 0.22023492,  0.22508108],
                                          [ 0.32595556,  0.11314971],
                                          [ 0.30494943,  0.19745198],
                                          [ 0.46857447,  0.21550114],
                                          [ 0.20011235,  0.5682324 ]],
                                         dtype=np.float32))
        self.embeddings = embeddings
        self.W = W
        self.b = b
        self.W_track = W_track

        def compose_network(inp, hidden, *args, **kwargs):
            if inp.ndim == 1:
                inp = inp[np.newaxis, :]
            if hidden.ndim == 1:
                hidden = hidden[np.newaxis, :]
            # TODO maybe can just change the `axis` flag in the above case?
            conc = T.concatenate([hidden, inp], axis=1)
            return (T.dot(conc, W) + b).squeeze()
        def track_network(state, inp, *args, **kwargs):
            conc = T.concatenate([state, inp], axis=1)
            state = T.dot(conc, W_track)
            logits = 0.0
            return state, logits

        def ghost_compose_net(c1, c2, buf_top, hidden):
            if c1.ndim == 1: c1 = c1[np.newaxis, :]
            if c2.ndim == 1: c2 = c2[np.newaxis, :]
            if buf_top.ndim == 1: buf_top = buf_top[np.newaxis, :]
            if hidden.ndim == 1: hidden = hidden[np.newaxis, :]

            inp_state = T.concatenate([c1, c2, buf_top], axis=1)
            hidden_next, _ = track_network(hidden, inp_state)

            comp = compose_network(T.concatenate([c1, c2], axis=1), hidden_next[:, self.model_dim / 2])

            return comp.squeeze(), hidden_next.squeeze()


        self.compose_network = compose_network
        self.track_network = track_network
        self.ghost_compose_net = ghost_compose_net

        self.X = T.imatrix("X")
        self.transitions = T.imatrix("transitions")
        self.y = T.ivector("y")

        self.stack = HardStack(
            self.model_dim, self.embedding_dim, self.batch_size,
            self.vocab_size, self.seq_length, self.compose_network,
            IdentityLayer, 0.0, 1.0, VariableStore(),
            prediction_and_tracking_network=self.track_network,
            use_tracking_lstm=True,
            tracking_lstm_hidden_dim=1,
            connect_tracking_comp=True,
            X=self.X,
            transitions=self.transitions,
            initial_embeddings=embeddings,
            use_input_batch_norm=False,
            use_input_dropout=False)

    def _fake_stack_ff(self):
        """Fake a stack feedforward S S M S M with the given data."""

        # seq_length * batch_size * emb_dim
        X_emb = self.stack.embeddings[self.X].dimshuffle(1, 0, 2)
        t0 = T.zeros((self.batch_size, self.model_dim))

        # Shift.
        tracking_inp = T.concatenate([T.zeros((self.batch_size, self.model_dim)),
                                      T.zeros((self.batch_size, self.model_dim)),
                                      X_emb[0]], axis=1)
        t1, _ = self.track_network(t0, tracking_inp)

        # Shift.
        tracking_inp = T.concatenate([X_emb[0],
                                      T.zeros((self.batch_size, self.model_dim)),
                                      X_emb[1]], axis=1)
        t2, _ = self.track_network(t1, tracking_inp)

        # Merge.
        tracking_inp = T.concatenate([X_emb[1], X_emb[0], X_emb[2]], axis=1)
        t3, _ = self.track_network(t2, tracking_inp)
        z3 = T.concatenate([X_emb[1], X_emb[0]], axis=1)
        c3 = self.compose_network(z3, t3[:, :self.model_dim / 2])

        # Shift.
        tracking_inp = T.concatenate([c3, T.zeros((self.batch_size, self.model_dim)), # TODO are we sure this is the value of stack_2 in thin-stack?
                                      X_emb[2]], axis=1)
        t4, _ = self.track_network(t3, tracking_inp)

        # Merge.
        tracking_inp = T.concatenate([X_emb[2], c3, X_emb[3]], axis=1)
        t5, _ = self.track_network(t4, tracking_inp)
        self.t5 = t5
        z5 = T.concatenate([X_emb[2], c3], axis=1)
        c5 = self.compose_network(z5, t5[:, :self.model_dim / 2])

        return c5

    def _make_cost(self, stack_top):
        logits = stack_top[:, :self.num_classes]
        costs = T.nnet.categorical_crossentropy(T.nnet.softmax(logits), self.y)
        return costs.mean()

    def test_backprop(self):
        # Simulate a batch of two token sequences, each with the same
        # transition sequence
        X = np.array([[0, 1, 2, 3, 1], [2, 1, 3, 0, 1]], dtype=np.int32)
        y = np.array([1, 0], dtype=np.int32)
        transitions = np.tile([0, 0, 1, 0, 1], (2, 1)).astype(np.int32)

        simulated_top = self._fake_stack_ff()
        simulated_cost = self._make_cost(simulated_top)
        f_simulated = theano.function(
            [self.X, self.y],
            (simulated_cost, T.grad(simulated_cost, self.W),
             T.grad(simulated_cost, self.b)))

        top = self.stack.final_stack[-self.batch_size:]
        cost = self._make_cost(top)
        error_signal = T.grad(cost, top)

        # Build composition gradient subgraph.
        f_delta = batch_subgraph_gradients([1, 1, 1, 1], [self.W, self.b, self.W_track], self.ghost_compose_net)

        # Now build backprop, passing in our composition gradient.
        self.stack.make_backprop_scan(
            [self.stack.final_aux_stack], error_signal, f_delta,
            [self.W.get_value().shape, self.b.get_value().shape])
        f = theano.function(
            [self.X, self.transitions, self.y],
            (cost, self.stack.deltas[0], self.stack.deltas[1]))
        #theano.printing.debugprint(f.maker.fgraph.outputs[1])

        #print T.grad(simulated_cost, self.t5)

        b_cost_sim, b_dW_sim, b_db_sim = f_simulated(X, y)
        b_cost, b_dW, b_db = f(X, transitions, y)

        np.testing.assert_almost_equal(b_cost_sim, b_cost)
        np.testing.assert_almost_equal(b_dW_sim, b_dW)
        np.testing.assert_almost_equal(b_db_sim, b_db)




if __name__ == '__main__':
    unittest.main()
