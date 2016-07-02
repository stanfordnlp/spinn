"""
A naive Theano implementation of a stack whose elements are symbolic vector
values. This "fat stack" powers the "fat classifier" model, which supports
training and inference in all model configurations.

Of course, we sacrifice speed for this flexibility. Theano's symbolic
differentiation algorithm isn't friendly to this implementation, and
so it suffers from poor backpropagation runtime efficiency. It is also
relatively slow to compile.
"""

from functools import partial

import numpy as np
import theano

from theano import tensor as T
from spinn import util


def update_stack(stack_t, shift_value, reduce_value, mask, model_dim):
    """
    Compute the new value of the given stack.

    This performs stack shifts and reduces in parallel, and somewhat
    wastefully.  It accepts a precomputed reduce result (in `reduce_value`) and
    a precomputed shift value `shift` for all examples, and switches between
    the two outcomes based on the per-example value of `mask`.

    Args:
        stack_t: Current stack value
        shift_value: Batch of values to be shifted
        reduce_value: Batch of reduce results
        mask: Batch of booleans: 1 if reduce, 0 if shift
        model_dim: The dimension of shift_value and reduce_value.
    """

    # Build two copies of the stack batch: one where every stack has received
    # a shift op, and one where every stack has received a reduce op.

    # Copy 1: Shift.
    stack_s = T.set_subtensor(stack_t[:, 0, :model_dim], shift_value)
    stack_s = T.set_subtensor(stack_s[:, 1:], stack_t[:, :-1])

    # Copy 2: Reduce.
    stack_r = T.set_subtensor(stack_t[:, 0, :model_dim], reduce_value)
    stack_r = T.set_subtensor(stack_r[:, 1:-1], stack_t[:, 2:])

    # Make sure mask broadcasts over all dimensions after the first.
    mask = mask.dimshuffle(0, "x", "x")
    mask = T.cast(mask, dtype=theano.config.floatX)
    stack_next = mask * stack_r + (1. - mask) * stack_s

    return stack_next


class HardStack(object):
    """
    Implementation of the SPINN model using a naive stack representation.

    This model scans a sequence using a hard stack. It optionally predicts
    stack operations using an MLP, and can receive supervision on these
    predictions from some external parser which acts as the "ground truth"
    parser.

    All of the SPINN model variants described in the paper can be instantiated
    with this class. See the file `checkpoints/commands.sh` in this project
    directory for concrete examples of how to invoke the variants.
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, seq_length, compose_network,
                 embedding_projection_network, training_mode, ground_truth_transitions_visible, vs,
                 prediction_and_tracking_network=None,
                 predict_use_cell=False,
                 predict_transitions=False,
                 train_with_predicted_transitions=False,
                 interpolate=False,
                 X=None,
                 transitions=None,
                 initial_embeddings=None,
                 make_test_fn=False,
                 use_input_batch_norm=True,
                 use_input_dropout=True,
                 embedding_dropout_keep_rate=1.0,
                 ss_mask_gen=None,
                 ss_prob=0.0,
                 use_tracking_lstm=False,
                 tracking_lstm_hidden_dim=8,
                 connect_tracking_comp=False,
                 context_sensitive_shift=False,
                 context_sensitive_use_relu=False,
                 use_attention=False,
                 premise_stack_tops=None,
                 attention_unit=None,
                 is_hypothesis=False,
                 initialize_hyp_tracking_state=False,
                 premise_tracking_c_state_final=None):
        """
        Construct a HardStack.

        Args:
            model_dim: Dimensionality of token embeddings and stack values
            word_embedding_dim: dimension of the word embedding
            vocab_size: Number of unique tokens in vocabulary
            seq_length: Maximum sequence length which will be processed by this
              stack
            compose_network: Blocks-like function which accepts arguments
              `inp, inp_dim, outp_dim, vs, name` (see e.g. `util.Linear`).
              Given a Theano batch `inp` of dimension `batch_size * inp_dim`,
              returns a transformed Theano batch of dimension
              `batch_size * outp_dim`.
            embedding_projection_network: Same form as `compose_network`.
            training_mode: A Theano scalar indicating whether to act as a training model
              with dropout (1.0) or to act as an eval model with rescaling (0.0).
            ground_truth_transitions_visible: A Theano scalar. If set (1.0), allow the model access
              to ground truth transitions. This can be disabled at evaluation time to force Model 1
              (or 2S) to evaluate in the Model 2 style with predicted transitions. Has no effect
              on Model 0.
            vs: VariableStore instance for parameter storage
            prediction_and_tracking_network: Blocks-like function which either maps values
              `3 * model_dim` to `action_dim` or uses the more complex TrackingUnit template.
            predict_use_cell: Only applicable when using an LSTM tracking unit
                for prediction (when `predict_transitions` is `True`). If
                `True`, use both the LSTM cell and hidden value to make
                predictions; use just the hidden value otherwise.
            predict_transitions: If set, predict transitions. If not, the tracking LSTM may still
              be used for other purposes.
            train_with_predicted_transitions: If `True`, use the predictions from the model
              (rather than the ground-truth `transitions`) to perform stack
              operations
            interpolate: If True, use scheduled sampling while training
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
            transitions: Theano batch describing transition matrix, or `None`
              (in which case this instance will make its own batch variable).
            initial_embeddings: pretrained embeddings or None
            make_test_fn: If set, create a function to run a scan for testing.
            use_input_batch_norm: If True, use batch normalization
            use_input_dropout: If True, use dropout
            embedding_dropout_keep_rate: The keep rate for dropout on projected embeddings.
            ss_mask_gen: A theano random stream
            ss_prob: Scheduled sampling probability
            use_tracking_lstm: If True, LSTM will be used in the tracking unit
            tracking_lstm_hidden_dim: hidden state dimension of the tracking LSTM
            connect_tracking_comp: If True, the hidden state of tracking LSTM will be
                fed to the TreeLSTM in the composition unit
            context_sensitive_shift: If True, the hidden state of tracking LSTM and the embedding
                vector will be used to calculate the vector that will be pushed onto the stack
            context_sensitive_use_relu: If True, a ReLU layer will be used while doing context
                sensitive shift, otherwise a Linear layer will be used
            use_attention: Use attention over premise tree nodes to obtain sentence representation (SNLI)
            premise_stack_tops: Tokens located on the top of premise stack. Used only when use_attention
                is set to True (SNLI)
            attention_unit: Function to implement the recurrent attention unit.
                Takes in the current attention state, current hypothesis stack top, all premise stack tops
                and returns the next attention state
            is_hypothesis: Whether we're processing the premise or the hypothesis (for SNLI)
            initialize_hyp_tracking_state: Initialize the c state of the tracking unit of hypothesis
                model with the final tracking unit c state of the premise model.
            premise_tracking_c_state_final: The final c state of the tracking unit in premise model.
        """

        self.model_dim = model_dim
        self.stack_dim = 2 * model_dim if use_attention in {"TreeWangJiang", "TreeThang"} else model_dim
        self.word_embedding_dim = word_embedding_dim
        self.use_tracking_lstm = use_tracking_lstm
        self.tracking_lstm_hidden_dim = tracking_lstm_hidden_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length

        self._compose_network = compose_network
        self._embedding_projection_network = embedding_projection_network
        self._prediction_and_tracking_network = prediction_and_tracking_network
        self._predict_use_cell = predict_use_cell
        self._predict_transitions = predict_transitions
        self.train_with_predicted_transitions = train_with_predicted_transitions

        self._vs = vs

        self.initial_embeddings = initial_embeddings

        self.training_mode = training_mode
        self.ground_truth_transitions_visible = ground_truth_transitions_visible
        self.embedding_dropout_keep_rate = embedding_dropout_keep_rate

        self.X = X
        self.transitions = transitions

        self.use_input_batch_norm = use_input_batch_norm
        self.use_input_dropout = use_input_dropout

        # Mask for scheduled sampling.
        self.ss_mask_gen = ss_mask_gen
        # Flag for scheduled sampling.
        self.interpolate = interpolate
        # Training step number.
        self.ss_prob = ss_prob
        # Connect tracking unit and composition unit.
        self.connect_tracking_comp = connect_tracking_comp
        assert (use_tracking_lstm or not connect_tracking_comp), \
            "Must use tracking LSTM if connecting tracking and composition units"
        self.context_sensitive_shift = context_sensitive_shift
        assert (use_tracking_lstm or not context_sensitive_shift), \
            "Must use tracking LSTM while doing context sensitive shift"
        self.context_sensitive_use_relu = context_sensitive_use_relu
        # Use constituent-by-constituent attention - true for both premise and hypothesis stacks
        self.use_attention = use_attention
        # Stores premise stack tops; none for the premise stack
        self.premise_stack_tops = premise_stack_tops
        # Attention unit
        if use_attention == "Rocktaschel" and is_hypothesis:
            self._attention_unit = util.RocktaschelAttentionUnit
        elif use_attention == "WangJiang" and is_hypothesis:
            self._attention_unit = util.WangJiangAttentionUnit
        elif use_attention == "TreeWangJiang" and is_hypothesis:
            self._attention_unit = util.TreeWangJiangAttentionUnit
        elif use_attention == "Thang" and is_hypothesis:
            self._attention_unit = util.ThangAttentionUnit
        elif use_attention == "TreeThang" and is_hypothesis:
            self._attention_unit = util.TreeThangAttentionUnit
        else:
            self._attention_unit = None
        self.initialize_hyp_tracking_state = initialize_hyp_tracking_state
        self.premise_tracking_c_state_final = premise_tracking_c_state_final
        if initialize_hyp_tracking_state:
            assert not is_hypothesis or premise_tracking_c_state_final is not None, \
                "Must supply initial c states in hypothesis model"

        # Check whether we're processing the hypothesis or the premise
        self.is_hypothesis = is_hypothesis

        self._make_params()
        self._make_inputs()
        self._make_scan()

        if make_test_fn:
            self.scan_fn = theano.function([self.X, self.transitions, self.training_mode,
                                            self.ground_truth_transitions_visible],
                                           self.final_stack,
                                           on_unused_input='warn')

    def _make_params(self):
        # Per-token embeddings.
        if self.initial_embeddings is not None:
            def EmbeddingInitializer(shape):
                return self.initial_embeddings
            self.word_embeddings = self._vs.add_param(
                    "embeddings", (self.vocab_size, self.word_embedding_dim),
                    initializer=EmbeddingInitializer,
                    trainable=False,
                    savable=False)
        else:
            self.word_embeddings = self._vs.add_param(
                "embeddings", (self.vocab_size, self.word_embedding_dim))

    def _make_inputs(self):
        self.X = self.X or T.imatrix("X")
        self.transitions = self.transitions or T.imatrix("transitions")

    def _step(self, transitions_t, ss_mask_gen_matrix_t, stack_t, buffer_cur_t,
            tracking_hidden, attention_hidden, buffer,
            ground_truth_transitions_visible, premise_stack_tops, projected_stack_tops):
        """TODO document"""
        batch_size, _ = self.X.shape

        # Extract top buffer values.
        idxs = buffer_cur_t + (T.arange(batch_size) * self.seq_length)

        if self.context_sensitive_shift:
            # Combine with the hidden state from previous unit.
            tracking_h_t = tracking_hidden[:, :self.tracking_lstm_hidden_dim]
            context_comb_input_t = T.concatenate([tracking_h_t, buffer[idxs]], axis=1)
            context_comb_input_dim = self.word_embedding_dim + self.tracking_lstm_hidden_dim
            comb_layer = util.ReLULayer if self.context_sensitive_use_relu else util.Linear
            buffer_top_t = comb_layer(context_comb_input_t, context_comb_input_dim, self.model_dim,
                                self._vs, name="context_comb_unit", use_bias=True,
                                initializer=util.HeKaimingInitializer())
        else:
            buffer_top_t = buffer[idxs]

        if self._prediction_and_tracking_network is not None:
            # We are predicting our own stack operations.
            h_dim = self.model_dim / 2 # TODO(SB): Turn this off when not using TreeLSTM.

            predict_inp = T.concatenate(
                [stack_t[:, 0, :h_dim], stack_t[:, 1, :h_dim], buffer_top_t[:, :h_dim]], axis=1)

            if self.use_tracking_lstm:
                # Update the hidden state and obtain predicted actions.
                tracking_hidden, actions_t = self._prediction_and_tracking_network(
                    tracking_hidden, predict_inp, h_dim * 3,
                    self.tracking_lstm_hidden_dim, self._vs,
                    logits_use_cell=self._predict_use_cell,
                    name="prediction_and_tracking")
            else:
                # Obtain predicted actions directly.
                actions_t = self._prediction_and_tracking_network(
                    predict_inp, h_dim * 3, util.NUM_TRANSITION_TYPES, self._vs,
                    logits_use_cell=self._predict_use_cell,
                    name="prediction_and_tracking")

        if self.train_with_predicted_transitions:
            # Model 2 case.
            if self.interpolate:
                # Only use ground truth transitions if they are marked as visible to the model.
                effective_ss_mask_gen_matrix_t = ss_mask_gen_matrix_t * ground_truth_transitions_visible
                # Interpolate between truth and prediction using bernoulli RVs
                # generated prior to the step.
                mask = (transitions_t * effective_ss_mask_gen_matrix_t
                        + actions_t.argmax(axis=1) * (1 - effective_ss_mask_gen_matrix_t))
            else:
                # Use predicted actions to build a mask.
                mask = actions_t.argmax(axis=1)
        elif self._predict_transitions:
            # Use transitions provided from external parser when not masked out
            mask = (transitions_t * ground_truth_transitions_visible
                        + actions_t.argmax(axis=1) * (1 - ground_truth_transitions_visible))
        else:
            # Model 0 case.
            mask = transitions_t

        # Now update the stack: first precompute reduce results.
        if self.model_dim != self.stack_dim:
            stack1 = stack_t[:, 0, :self.model_dim].reshape((-1, self.model_dim))
            stack2 = stack_t[:, 1, :self.model_dim].reshape((-1, self.model_dim))
        else:
            stack1 = stack_t[:, 0].reshape((-1, self.model_dim))
            stack2 = stack_t[:, 1].reshape((-1, self.model_dim))
        reduce_items = (stack1, stack2)
        if self.connect_tracking_comp:
            tracking_h_t = tracking_hidden[:, :self.tracking_lstm_hidden_dim]
            reduce_value = self._compose_network(reduce_items, tracking_h_t, self.model_dim,
                self._vs, name="compose", external_state_dim=self.tracking_lstm_hidden_dim)
        else:
            reduce_value = self._compose_network(reduce_items, (self.model_dim,) * 2, self.model_dim,
                self._vs, name="compose")

        # Compute new stack value.
        stack_next = update_stack(stack_t, buffer_top_t, reduce_value, mask,
                                  self.model_dim)

        # If attention is to be used and premise_stack_tops is not None (i.e.
        # we're processing the hypothesis) calculate the attention weighed representation.
        if self.use_attention != "None" and self.is_hypothesis:
            h_dim = self.model_dim / 2
            if self.use_attention in {"TreeWangJiang", "TreeThang"}:
                mask_ = mask.dimshuffle(0, "x")
                mask_ = T.cast(mask_, dtype=theano.config.floatX)
                attention_hidden_l = stack_t[:, 0, self.model_dim:] * mask_
                attention_hidden_r = stack_t[:, 1, self.model_dim:] * mask_

                tree_attention_hidden = self._attention_unit(attention_hidden_l, attention_hidden_r,
                    stack_next[:, 0, :h_dim], premise_stack_tops, projected_stack_tops, h_dim,
                    self._vs, name="attention_unit")
                stack_next = T.set_subtensor(stack_next[:, 0, self.model_dim:], tree_attention_hidden)
            else:
                attention_hidden = self._attention_unit(attention_hidden, stack_next[:, 0, :h_dim],
                    premise_stack_tops, projected_stack_tops, h_dim, self._vs, name="attention_unit")

        # Move buffer cursor as necessary. Since mask == 1 when reduce, we
        # should increment each buffer cursor by 1 - mask.
        buffer_cur_next = buffer_cur_t + (1 - mask)

        if self._predict_transitions:
            ret_val = stack_next, buffer_cur_next, tracking_hidden, attention_hidden, actions_t
        else:
            ret_val = stack_next, buffer_cur_next, tracking_hidden, attention_hidden

        if not self.interpolate:
            # Use ss_mask as a redundant return value.
            ret_val = (ss_mask_gen_matrix_t,) + ret_val

        return ret_val

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_size, max_stack_size = self.X.shape

        # Stack batch is a 3D tensor.
        stack_shape = (batch_size, max_stack_size, self.stack_dim)
        stack_init = T.zeros(stack_shape)

        # Look up all of the embeddings that will be used.
        raw_embeddings = self.word_embeddings[self.X]  # batch_size * seq_length * emb_dim

        if self.context_sensitive_shift:
            # Use the raw embedding vectors, they will be combined with the hidden state of
            # the tracking unit later
            buffer_t = raw_embeddings
            buffer_emb_dim = self.word_embedding_dim
        else:
            # Allocate a "buffer" stack initialized with projected embeddings,
            # and maintain a cursor in this buffer.
            if self.use_input_dropout:
                raw_embeddings = util.Dropout(raw_embeddings, self.embedding_dropout_keep_rate, self.training_mode)
            buffer_t = self._embedding_projection_network(
                raw_embeddings, self.word_embedding_dim, self.model_dim, self._vs, name="project")
            if self.use_input_batch_norm:
                buffer_t = util.BatchNorm(buffer_t, self.model_dim, self._vs, "buffer",
                    self.training_mode, axes=[0, 1])
            buffer_emb_dim = self.model_dim

        # Collapse buffer to (batch_size * buffer_size) * emb_dim for fast indexing.
        buffer_t = buffer_t.reshape((-1, buffer_emb_dim))

        buffer_cur_init = T.zeros((batch_size,), dtype="int")

        DUMMY = T.zeros((2,)) # a dummy tensor used as a place-holder

        # Dimshuffle inputs to seq_len * batch_size for scanning
        transitions = self.transitions.dimshuffle(1, 0)

        # Initialize the hidden state for the tracking LSTM, if needed.
        if self.use_tracking_lstm:
            if self.initialize_hyp_tracking_state and self.is_hypothesis:
                # Initialize the c state of tracking unit from the c state of premise model.
                h_state_init = T.zeros((batch_size, self.tracking_lstm_hidden_dim))
                hidden_init = T.concatenate([h_state_init, self.premise_tracking_c_state_final], axis=1)
            else:
                hidden_init = T.zeros((batch_size, self.tracking_lstm_hidden_dim * 2))
        else:
            hidden_init = DUMMY

        # Initialize the attention representation if needed
        if self.use_attention not in {"TreeWangJiang", "TreeThang", "None"} and self.is_hypothesis:
            h_dim = self.model_dim / 2
            if self.use_attention == "WangJiang" or self.use_attention == "Thang":
                 attention_init = T.zeros((batch_size, 2 * h_dim))
            else:
                attention_init = T.zeros((batch_size, h_dim))
        else:
            # If we're not using a sequential attention accumulator (i.e., no attention or
            # tree attention), use a size-zero value here.
            attention_init = DUMMY

        # Set up the output list for scanning over _step().
        if self._predict_transitions:
            outputs_info = [stack_init, buffer_cur_init, hidden_init, attention_init, None]
        else:
            outputs_info = [stack_init, buffer_cur_init, hidden_init, attention_init]

        # Prepare data to scan over.
        sequences = [transitions]
        if self.interpolate:
            # Generate Bernoulli RVs to simulate scheduled sampling
            # if the interpolate flag is on.
            ss_mask_gen_matrix = self.ss_mask_gen.binomial(
                                transitions.shape, p=self.ss_prob)
            # Take in the RV sequence as input.
            sequences.append(ss_mask_gen_matrix)
        else:
            # Take in the RV sequqnce as a dummy output. This is
            # done to avaid defining another step function.
            outputs_info = [DUMMY] + outputs_info

        non_sequences = [buffer_t, self.ground_truth_transitions_visible]

        if self.use_attention != "None" and self.is_hypothesis:
            h_dim = self.model_dim / 2
            projected_stack_tops = util.AttentionUnitInit(self.premise_stack_tops, h_dim, self._vs)
            non_sequences = non_sequences + [self.premise_stack_tops, projected_stack_tops]
        else:
            DUMMY2 = T.zeros((2,)) # another dummy tensor
            non_sequences = non_sequences + [DUMMY, DUMMY2]

        scan_ret = theano.scan(
                self._step,
                sequences=sequences,
                non_sequences=non_sequences,
                outputs_info=outputs_info,
                n_steps=self.seq_length,
                name="stack_fwd")

        stack_ind = 0 if self.interpolate else 1
        self.final_stack = scan_ret[0][stack_ind][-1]
        self.final_representations = self.final_stack[:, 0, :self.model_dim]
        self.embeddings = self.final_stack[:, 0]

        if self._predict_transitions:
            self.transitions_pred = scan_ret[0][-1].dimshuffle(1, 0, 2)
        else:
            self.transitions_pred = T.zeros((batch_size, 0))


        if self.use_attention != "None" and not self.is_hypothesis:
            # Store the stack top at each step as an attribute.
            h_dim = self.model_dim / 2
            self.stack_tops = scan_ret[0][stack_ind][:,:,0,:h_dim].reshape((max_stack_size, batch_size, h_dim))

        if self.use_attention != "None" and self.is_hypothesis:
            h_dim = self.model_dim / 2
            if self.use_attention == "Rocktaschel":
                self.final_weighed_representation = util.AttentionUnitFinalRepresentation(scan_ret[0][stack_ind + 3][-1],
                    self.embeddings[:,:h_dim], h_dim, self._vs)
            elif self.use_attention in {"WangJiang", "Thang"}:
                self.final_weighed_representation = scan_ret[0][stack_ind+3][-1][:,:h_dim]
            elif self.use_attention in {"TreeWangJiang", "TreeThang"}:
                self.final_weighed_representation = scan_ret[0][stack_ind][-1][:,0,2*h_dim:3*h_dim]

        if self.initialize_hyp_tracking_state and not self.is_hypothesis:
            # Store the final c states of the tracking unit.
            self.tracking_c_state_final = scan_ret[0][stack_ind+2][-1][:, self.tracking_lstm_hidden_dim:]
        else:
            self.tracking_c_state_final = None

class Model0(HardStack):

    def __init__(self, *args, **kwargs):
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = partial(util.TrackingUnit, make_logits=False)
        else:
            kwargs["prediction_and_tracking_network"] = None

        kwargs["predict_transitions"] = False
        kwargs["train_with_predicted_transitions"] = False
        kwargs["interpolate"] = False
        super(Model0, self).__init__(*args, **kwargs)


class Model1(HardStack):

    def __init__(self, *args, **kwargs):
        # Set the tracking unit based on supplied tracking_lstm_hidden_dim.
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = util.TrackingUnit
        else:
            kwargs["prediction_and_tracking_network"] = util.Linear
        # Defaults to not using predictions while training and not using scheduled sampling.
        kwargs["predict_transitions"] = True
        kwargs["train_with_predicted_transitions"] = False
        super(Model1, self).__init__(*args, **kwargs)


class Model2(HardStack):

    def __init__(self, *args, **kwargs):
        # Set the tracking unit based on supplied tracking_lstm_hidden_dim.
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = util.TrackingUnit
        else:
            kwargs["prediction_and_tracking_network"] = util.Linear
        # Defaults to using predictions while training and not using scheduled sampling.
        kwargs["predict_transitions"] = True
        kwargs["train_with_predicted_transitions"] = True
        kwargs["interpolate"] = False
        super(Model2, self).__init__(*args, **kwargs)


class Model2S(HardStack):

    def __init__(self, *args, **kwargs):
        use_tracking_lstm = kwargs.get("use_tracking_lstm", False)
        if use_tracking_lstm:
            kwargs["prediction_and_tracking_network"] = util.TrackingUnit
        else:
            kwargs["prediction_and_tracking_network"] = util.Linear
        # Use supplied settings and use scheduled sampling.
        kwargs["predict_transitions"] = True
        kwargs["train_with_predicted_transitions"] = True
        kwargs["interpolate"] = True
        super(Model2S, self).__init__(*args, **kwargs)
