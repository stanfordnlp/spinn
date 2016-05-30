"""Theano-based stack implementations."""

from functools import partial

import numpy as np
import theano
from theano.ifelse import ifelse

from theano import tensor as T
from spinn import util
from spinn.util import cuda


def update_hard_stack(t, t_f, stack_t, shift_value, reduce_value, queue_t,
                      cursors_t, mask, batch_size, stack_shift, cursors_shift):
    """Compute the new value of the given hard stack.

    This performs stack shifts and reduces in parallel, and somewhat
    wastefully.  It accepts a precomputed reduce result (in `reduce_value`) and
    a precomputed shift value `shift` for all examples, and switches between
    the two outcomes based on the per-example value of `mask`.

    Args:
        t: (int scalar) current timestep
        t_f: (float scalar) current timestep
        shift_value: (float batch) New stack-tops for each example, if each
            example were to shift at this timestep
        reduce_value: (float batch) New stack-tops for each example, if each
            example were to reduce at this timestep
        queue_t: Aux queue after previous operation
        cursors_t: Queue cursors after previous operation
        mask: (float batch) 0 or 1 for each example; 0 if shift, 1 if reduce
        batch_size:
        stack_shift: ThinStack._stack_shift
        cursors_shift: ThinStack._cursors_shift
    """

    # Implementation of the core thin-stack update.

    mask2 = mask.dimshuffle(0, "x")
    top_next = mask2 * reduce_value + (1 - mask2) * shift_value
    stack_next = cuda.AdvancedIncSubtensor1Floats(set_instead_of_inc=True, inplace=True)(
            stack_t, top_next, t_f * batch_size + stack_shift)

    cursors_next = cursors_t + (mask * -1 + (1 - mask) * 1)
    queue_next = cuda.AdvancedIncSubtensor1Floats(set_instead_of_inc=True, inplace=True)(
            queue_t, t_f, cursors_shift + cursors_next)

    return stack_next, queue_next, cursors_next


class ThinStack(object):
    """
    Main implementation of the thin-stack algorithm.

    This model scans a sequence using a hard stack, unrolling the given stack
    recurrence over the entire sequence. The recurrence can optionally predict
    stack operations using an MLP, and can receive supervision on these
    predictions from some external parser which acts as the "ground truth"
    parser.
    """

    def __init__(self, spec, recurrence, embedding_projection_network,
                 training_mode, ground_truth_transitions_visible, vs,
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
                 is_hypothesis=False,
                 name="stack"):
        """
        Construct a ThinStack.

        Args:
            spec: `ModelSpec` instance.
            recurrence: `Recurrence` instance which specifies the stack
                recurrence to be unfolded over time.
            embedding_projection_network: Same form as `compose_network`.
            training_mode: A Theano scalar indicating whether to act as a training model
              with dropout (1.0) or to act as an eval model with rescaling (0.0).
            ground_truth_transitions_visible: A Theano scalar. If set (1.0), allow the model access
              to ground truth transitions. This can be disabled at evaluation time to force Model 1
              (or 2S) to evaluate in the Model 2 style with predicted transitions. Has no effect
              on Model 0.
            vs: VariableStore instance for parameter storage
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
            is_hypothesis: Whether we're processing the premise or the hypothesis (for SNLI)
        """

        self.spec = spec
        self.recurrence = recurrence

        self.batch_size = spec.batch_size
        self.vocab_size = spec.vocab_size
        self.seq_length = spec.seq_length
        self.stack_size = self.seq_length + 1
        self.model_dim = spec.model_dim
        self.stack_dim = self.model_dim
        self.word_embedding_dim = spec.word_embedding_dim

        self._embedding_projection_network = embedding_projection_network
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

        # Check whether we're processing the hypothesis or the premise
        self.is_hypothesis = is_hypothesis

        self.name = name
        self._prefix = "" if name is None else name + "/"

        # Track variable store contents before / after graph construction.
        vars_before = set(self._vs.vars.keys())

        self._make_params()
        self._make_shared()
        self._make_inputs()
        self._make_scan()

        self._vars = set(self._vs.vars.keys()) - vars_before

        if make_test_fn:
            scan_fn = theano.function([self.X, self.transitions, self.training_mode,
                                       self.ground_truth_transitions_visible],
                                   self.stack, updates=self.scan_updates,
                                   accept_inplace=True,
                                   on_unused_input="warn")
            def scan_fn_wrapper(*args):
                scan_fn(*args)
                ret = self.stack.get_value()
                return ret[self.batch_size:]
            self.scan_fn = scan_fn_wrapper

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

    def _make_shared(self):
        # Build main stack. This stores the shift/reduce outputs at each
        # timestep.
        stack_shape = (self.stack_size * self.batch_size, self.model_dim)
        stack_init = np.zeros(stack_shape, dtype=np.float32)
        self.stack = theano.shared(stack_init, borrow=False,
                                   name=self._prefix + "stack")

        # Build backward stack. This stores d(cost)/d(stack value) at each
        # timestep.
        stack_bwd_init = np.zeros(stack_shape, dtype=np.float32)
        self.stack_bwd = theano.shared(stack_bwd_init, borrow=False,
                                       name=self._prefix + "stack_bwd")

        # Build auxiliary stacks for extra recurrence outputs.
        aux_stack_shapes = [(self.stack_size * self.batch_size,) + shape
                            for shape in self.recurrence.extra_outputs]
        self.aux_stacks = [theano.shared(np.zeros(shape, dtype=np.float32),
                                         name=self._prefix + "aux_stack_%i" % i)
                           for i, shape in enumerate(aux_stack_shapes)]

        # Build backward auxiliary stacks. These store d(cost)/d(auxiliary) at
        # each timestep.
        self.aux_bwd_stacks = [theano.shared(np.zeros(shape, dtype=np.float32),
                                             name=self._prefix + "aux_bwd_stack_%i" % i)
                               for i, shape in enumerate(aux_stack_shapes)]

        # Build cursor vector.
        cursors_init = np.zeros((self.batch_size,), dtype=np.float32) + self.seq_length
        self.cursors = theano.shared(cursors_init, borrow=False,
                                     name=self._prefix + "cursors")

        # Build queue matrix.
        queue_init = np.zeros((self.batch_size * ((2 * self.seq_length) + 1),),
                              dtype=np.float32)
        self.queue = theano.shared(queue_init, borrow=False,
                                   name=self._prefix + "queue")

        all_vars = [self.stack, self.stack_bwd, self.cursors, self.queue]
        all_vars += self.aux_stacks + self.aux_bwd_stacks

        # Track which shared variables need to be cleared on each batch.
        self._zero_updates = all_vars

        # Don't compile our `zero` function until last-minute. This lets the
        # backprop unrolling (if used) add its own data to `_zero_updates`.
        self._zero = None

    def zero(self):
        if self._zero is None:
            # JIT-prepare the zero function.
            zero_updates = {var: np.zeros(var.get_value().shape,
                                          dtype=np.float32)
                            for var in self._zero_updates}
            self._zero = theano.function([], [], updates=zero_updates)

        self._zero()

    def _make_inputs(self):
        self.X = self.X or T.imatrix(self._prefix + "X")
        self.transitions = self.transitions or T.imatrix(self._prefix + "transitions")

    def _step(self, t, t_f, transitions_t, transitions_t_f, ss_mask_gen_matrix_t,
              buffer_cur_t, buffer, ground_truth_transitions_visible, *rest):
        batch_size = self.batch_size

        # Extract top buffer values.
        idxs = buffer_cur_t + self._buffer_shift
        buffer_top_t = cuda.AdvancedSubtensor1Floats("F_buffer_top")(buffer, idxs)

        # Fetch top two stack elements.
        stack_1_ptrs = (t - 1) * self.batch_size + self._stack_shift
        stack_1 = cuda.AdvancedSubtensor1Floats("F_stack1")(self.stack, stack_1_ptrs)

        # Get pointers into stack for second-to-top element.
        cursors = self.cursors - 1.0
        stack_2_ptrs = cuda.AdvancedSubtensor1Floats("F_stack2_ptrs")(self.queue, cursors + self._queue_shift)
        stack_2_ptrs = stack_2_ptrs * batch_size + self._stack_shift

        # Retrieve second-to-top element.
        stack_2 = cuda.AdvancedSubtensor1Floats("F_stack2")(self.stack, stack_2_ptrs)

        extra_inputs = tuple([
            cuda.AdvancedSubtensor1Floats("F_extra_inp_%i" % i)(
                aux_stack, stack_1_ptrs)
            for i, aux_stack in enumerate(self.aux_stacks)])

        recurrence_inputs = (stack_1, stack_2, buffer_top_t)
        recurrence_inputs += extra_inputs
        recurrence_ret = self.recurrence(recurrence_inputs)

        shift_ret, reduce_ret = recurrence_ret[:2]

        if self.recurrence.predicts_transitions:
            actions_t = recurrence_ret[2].argmax(axis=1)

            if self.recurrence.uses_predictions:

                # Model 2 case.
                if self.interpolate:
                    # Only use ground truth transitions if they are marked as visible to the model.
                    effective_ss_mask_gen_matrix_t = ss_mask_gen_matrix_t * ground_truth_transitions_visible

                    # Interpolate between truth and prediction using bernoulli RVs.
                    # generated prior to the step.
                    mask = (transitions_t * effective_ss_mask_gen_matrix_t
                            + actions_t * (1 - effective_ss_mask_gen_matrix_t))
                else:
                    # Use predicted actions to build a mask.
                    mask = actions_t
            else:
                # Use transitions provided from external parser when not masked out.
                mask = (transitions_t * ground_truth_transitions_visible
                        + actions_t * (1 - ground_truth_transitions_visible))
        else:
            # Model 0 case.
            mask = transitions_t_f

        # Compute new stack value.
        stack_next, queue_next, cursors_next = update_hard_stack(
            t, t_f, self.stack, buffer_top_t, reduce_ret[0], self.queue, self.cursors,
            mask, self.batch_size, self._stack_shift, self._cursors_shift)

        # Move buffer cursor as necessary. Since mask == 1 when reduce, we
        # should increment each buffer cursor by 1 - mask.
        buffer_cur_next = buffer_cur_t + (1 - transitions_t_f)

        # Update auxiliary stacks.
        mask2 = mask.dimshuffle(0, "x")
        ptr_next = t_f * self.batch_size + self._stack_shift
        # TODO: Allow recurrence to state that output i is the same for shift
        # and reduce -- then we don't have to mask it
        aux_outputs = [mask2 * r_output + (1. - mask2) * s_output
                       for s_output, r_output in zip(shift_ret, reduce_ret[1:])]
        aux_stack_updates = {aux_stack: cuda.AdvancedIncSubtensor1Floats(set_instead_of_inc=True, inplace=True)(
            aux_stack, aux_output, ptr_next)
            for aux_stack, aux_output in zip(self.aux_stacks, aux_outputs)}

        if self.recurrence.predicts_transitions:
            ret_val = buffer_cur_next, stack_2_ptrs, actions_t
        else:
            ret_val = buffer_cur_next, stack_2_ptrs

        if not self.interpolate:
            # Use ss_mask as a redundant return value.
            ret_val = (ss_mask_gen_matrix_t,) + ret_val

        updates = {
            self.stack: stack_next,
            self.queue: queue_next,
            self.cursors: cursors_next
        }
        updates.update(aux_stack_updates)

        return ret_val, updates

    def _project_embeddings(self, raw_embeddings, dropout_mask=None):
        """
        Run a forward pass of the embedding projection network, retaining
        intermediate values in order to support backpropagation.
        """
        projected = self._embedding_projection_network(
            raw_embeddings, self.word_embedding_dim, self.model_dim, self._vs,
            name=self._prefix + "project")

        if self.use_input_batch_norm:
            projected = util.BatchNorm(
                projected, self.model_dim, self._vs, self._prefix + "buffer",
                self.training_mode, axes=[0, 1])

        # Dropout.
        # If we use dropout, we need to retain the mask for backprop purposes.
        ret_dropout_mask = None
        if self.use_input_dropout:
            projected, ret_dropout_mask = util.Dropout(
                projected, self.embedding_dropout_keep_rate,
                self.training_mode, dropout_mask=dropout_mask,
                return_mask=True)

        return projected, ret_dropout_mask

    def _make_scan(self):
        """Build the sequential composition / scan graph."""

        batch_size = self.batch_size
        max_stack_size = stack_size = self.stack_size
        self.batch_range = batch_range = T.arange(batch_size, dtype="int32")

        # Various indexing helpers. Make sure they're float32.
        self._queue_shift = T.cast(batch_range * self.seq_length,
                                   theano.config.floatX)
        self._buffer_shift = self._queue_shift
        self._cursors_shift = self._queue_shift
        self._stack_shift = T.cast(batch_range, theano.config.floatX)

        # Look up all of the embeddings that will be used.
        raw_embeddings = self.embeddings[self.X]  # batch_size * seq_length * emb_dim
        # Flatten for easy indexing.
        self._raw_buffer_t = raw_embeddings.reshape((-1, self.word_embedding_dim))

        # Allocate a "buffer" stack initialized with projected embeddings,
        # and maintain a cursor in this buffer.
        buffer_t, dropout_mask = self._project_embeddings(self._raw_buffer_t)
        self._embedding_dropout_masks = dropout_mask

        self.buffer_t = buffer_t

        buffer_cur_init = util.zeros_nobroadcast((batch_size,))

        DUMMY = T.zeros((2, 2)) # a dummy tensor used as a place-holder.

        # Dimshuffle inputs to seq_len * batch_size for scanning.
        transitions = self.transitions.dimshuffle(1, 0)
        transitions_f = T.cast(transitions, dtype=theano.config.floatX)

        # Set up the output list for scanning over _step().
        # Final `None` entry is for accumulating `stack_2_ptr` values.
        outputs_info = [buffer_cur_init, None]
        if self.recurrence.predicts_transitions:
            outputs_info.append(None)

        # Prepare data to scan over.
        sequences = [# timesteps
                     T.arange(1, self.seq_length + 1),
                     # timesteps (as float)
                     T.arange(1, self.seq_length + 1, dtype="float32"),
                     # input transitions (as int and as float)
                     transitions, transitions_f]
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

        # Tack on all relevant params, structures to satisfy strict mode.
        non_sequences += self.aux_stacks + self._vs.vars.values()

        scan_ret, self.scan_updates = theano.scan(
                self._step,
                sequences=sequences,
                non_sequences=non_sequences,
                outputs_info=outputs_info,
                #strict=True,
                name=self._prefix + "fwd")

        ret_shift = 0 if self.interpolate else 1
        self.final_buf = scan_ret[ret_shift + 0][-1]
        self.stack_2_ptrs = scan_ret[ret_shift + 1]
        self.buf_ptrs = scan_ret[ret_shift + 0]

        self.final_stack = self.scan_updates[self.stack]
        self.final_aux_stacks = [self.scan_updates[aux_stack]
                                 for aux_stack in self.aux_stacks]
        self.sentence_embeddings = self.final_stack[-self.batch_size:]

        self.transitions_pred = None
        if self.recurrence.predicts_transitions:
            self.transitions_pred = scan_ret[ret_shift + 2][-1].dimshuffle(1, 0, 2)

    def _make_backward_graphs(self):
        """Generate gradient subgraphs for this stack's recurrence."""

        input_ndim = [2] * 3
        input_ndim += [len(extra_output_shape) + 1
                       for extra_output_shape in self.recurrence.extra_outputs]

        wrt = [self._vs.vars[key] for key in self._vars
               if key != "embeddings" and key in self._vs.trainable_vars]

        # TODO handle gradient of action prediction
        # TODO would it pay off to force these to have the same concrete
        # subgraph instances rather than creating it twice?
        @util.ensure_2d_arguments
        def p_fwd(*inputs, **constants):
            return self.recurrence(inputs, **constants)[0]
        @util.ensure_2d_arguments
        def m_fwd(*inputs, **constants):
            return self.recurrence(inputs, **constants)[1]

        f_s_delta = util.batch_subgraph_gradients(input_ndim, wrt, p_fwd,
                                                  name=self._prefix + "bwd_graph_shift")
        f_r_delta = util.batch_subgraph_gradients(input_ndim, wrt, m_fwd,
                                                  name=self._prefix + "bwd_graph_reduce")

        # Also create a backprop graph for the embedding projection network.
        f_proj_delta = None
        if self._embedding_projection_network not in [None, util.IdentityLayer]:
            projection_ndim = [2]
            if self.use_input_dropout:
                projection_ndim += [2]
            f_proj_delta = util.batch_subgraph_gradients(projection_ndim, wrt,
                                                         self._project_embeddings,
                                                         name=self._prefix + "bwd_graph_proj")

        return wrt, f_proj_delta, f_s_delta, f_r_delta

    def make_backprop_scan(self, error_signal,
                           extra_cost_inputs=None,
                           compute_embedding_gradients=True):
        """
        Compile the backpropagation graph for a ThinStack instance.

        This method analyzes the feedforward graph of a ThinStack instance and
        analytically calculates its gradient with respect to the stack's
        parameters of interest.

        This method must be called by the client after constructing ThinStack.
        After creating the gradient graph, the outputs of the graph (gradient
        batches) are installed into an instance member self.gradients. This
        object is a dictionary mapping from stack shared variables (parameters)
        to their symbolic gradients.

        Args:
            error_signal: The external gradient d(cost)/d(stack top). A Theano
                batch of size `batch_size * model_dim`.
            extra_cost_inputs: Other symbolic variables which may be involved
                in the symbolic cost expression / the error signal that should
                be passed as extra inputs to a backpropagation scan.
            compute_embedding_gradients: Calculate gradients for each embedding
                vector.
        """

        # The meat of the implementation of this method is in the inner
        # function step_b. What comes before is mainly preparatory code; what
        # comes after is the scan invocation that unrolls the step_b function.

        assert hasattr(self, "stack_2_ptrs"), \
            ("self._make_scan (forward pass) must be defined before "
             "self.make_backprop_scan is called")

        # We need to add extra updates to the `_zero_updates` member, so we
        # must be called before `_zero_updates` is read.
        assert self._zero is None, \
            ("Can only install backprop on a fresh ThinStack. Don't call "
             "ThinStack.zero before setting up backprop.")

        if (compute_embedding_gradients
            and self._embedding_projection_network not in [None, util.IdentityLayer]):
            raise ValueError(
                "Do not support backprop for both an embedding projection "
                "layer and individual embeddings.")

        if self.use_input_batch_norm:
            raise ValueError(
                "Thin-stack backprop not supported with input batch-norm. Jon "
                "worked on BN gradients for 3 days without success, and then "
                "dropped it.")

        if extra_cost_inputs is None:
            extra_cost_inputs = []

        # Analytically calculate backpropagation graphs for an abstract single
        # timestep. The list 'wrt' contains all variables for which we should
        # collect gradients (this list of course depends on the nature of the
        # forward-prop activation function). The latter 3 returns are actually
        # graph "thunks" which return a concrete backpropagation graph when
        # provided with actual symbolic inputs / error signals.
        #
        # These graphs will be repeatedly invoked and chained together in the
        # code further below.
        wrt, f_proj_delta, f_shift_delta, f_reduce_delta = \
                self._make_backward_graphs()
        wrt_shapes = [wrt_i.get_value().shape for wrt_i in wrt]

        # Build shared variables for accumulating wrt deltas.
        wrt_vars = [theano.shared(np.zeros(wrt_shape, dtype=np.float32),
                                  name=self._prefix + "bwd/wrt/%s" % wrt_i)
                    for wrt_i, wrt_shape in zip(wrt, wrt_shapes)]
        # All of these need to be zeroed out in between batches.
        self._zero_updates += wrt_vars

        # Also accumulate embedding gradients separately
        if compute_embedding_gradients:
            dE = theano.shared(np.zeros(self.embeddings.get_value().shape,
                                        dtype=np.float32),
                               name=self._prefix + "bwd/wrt/embeddings")
            self._zero_updates.append(dE)
        else:
            # Make dE a dummy variable.
            dE = T.zeros((1,))

        # Useful batch zero-constants.
        zero_stack = T.zeros((self.batch_size, self.model_dim))
        zero_extra_inps = [T.zeros((self.batch_size, extra_shape[-1]))
                           for extra_shape in self.recurrence.extra_outputs]
        # Zero Jacobian matrices for masked reductions during backprop. May not
        # be used.
        zero_jac_wrts = [T.zeros((self.batch_size,) + wrt_shape)
                         for wrt_shape in wrt_shapes]

        DUMMY = util.zeros_nobroadcast((1,))

        def lookup(t_f, stack_fwd, stack_2_ptrs_t, buffer_cur_t,
                  stack_bwd_t, extra_bwd):
            """Retrieve all relevant bwd inputs/outputs at time `t`."""

            grad_cursor = t_f * self.batch_size + self._stack_shift
            main_grad = cuda.AdvancedSubtensor1Floats("B_maingrad")(
                stack_bwd_t, grad_cursor)
            extra_grads = tuple([
                cuda.AdvancedSubtensor1Floats("B_extragrad_%i" % i)(
                    extra_bwd_i, grad_cursor)
                for i, extra_bwd_i in enumerate(extra_bwd)])

            # Find the timesteps of the two elements involved in the potential
            # reduce at this timestep.
            t_c1 = (t_f - 1.0) * self.batch_size + self._stack_shift
            t_c2 = stack_2_ptrs_t

            # Find the two elements involved in the potential reduce.
            c1 = cuda.AdvancedSubtensor1Floats("B_stack1")(stack_fwd, t_c1)
            c2 = cuda.AdvancedSubtensor1Floats("B_stack2")(stack_fwd, t_c2)

            buffer_top_t = cuda.AdvancedSubtensor1Floats("B_buffer_top")(
                self.buffer_t, buffer_cur_t + self._buffer_shift)

            # Retrieve extra inputs from auxiliary stack(s).
            extra_inps_t = tuple([
                cuda.AdvancedSubtensor1Floats("B_extra_inp_%i" % i)(
                    extra_inp_i, t_c1)
                for extra_inp_i in self.final_aux_stacks])

            inputs = (c1, c2, buffer_top_t) + extra_inps_t
            grads = (main_grad,) + extra_grads
            return t_c1, t_c2, inputs, grads

        def step_b(# sequences
                   t_f, transitions_t_f, stack_2_ptrs_t, buffer_cur_t,
                   # accumulators
                   dE,
                   # rest (incl. outputs_info, non_sequences)
                   *rest):

            # Separate the accum arguments from the non-sequence arguments.
            n_wrt = len(wrt_shapes)
            n_extra_bwd = len(self.recurrence.extra_outputs)
            wrt_deltas = rest[:n_wrt]
            stack_bwd_t = rest[n_wrt]
            extra_bwd = rest[n_wrt + 1:n_wrt + 1 + n_extra_bwd]
            id_buffer, stack_final = \
                rest[n_wrt + 1 + n_extra_bwd:n_wrt + 1 + n_extra_bwd + 2]

            # At first iteration, drop the external error signal into the main
            # backward stack.
            stack_bwd_next = ifelse(T.eq(t_f, self.seq_length),
                                    T.set_subtensor(stack_bwd_t[-self.batch_size:], error_signal),
                                    stack_bwd_t)


            # Retrieve all relevant inputs/outputs at this timestep.
            t_c1, t_c2, inputs, grads = \
                lookup(t_f, stack_final, stack_2_ptrs_t, buffer_cur_t,
                       stack_bwd_next, extra_bwd)
            main_grad = grads[0]

            # Calculate deltas for this timestep.
            r_delta_inp, r_delta_wrt = f_reduce_delta(inputs, grads)
            # NB: main_grad is not passed to shift function.
            s_delta_inp, s_delta_wrt = f_shift_delta(inputs, grads[1:])

            # Check that delta function outputs match (at least in number).
            assert len(r_delta_inp) == len(s_delta_inp), \
                "%i %i" % (len(r_delta_inp), len(s_delta_inp))
            assert len(r_delta_wrt) == len(s_delta_wrt), \
                "%i %i" % (len(r_delta_wrt), len(s_delta_wrt))
            assert len(r_delta_inp) == 3 + len(self.aux_stacks), \
                "%i %i" % (len(r_delta_inp), 3 + len(self.aux_stacks))
            assert len(r_delta_wrt) == len(wrt)

            # Retrieve embedding indices on buffer at this timestep.
            # (Necessary for sending embedding gradients.)
            buffer_ids_t = cuda.AdvancedSubtensor1Floats("B_buffer_ids")(
                    id_buffer, buffer_cur_t + self._buffer_shift)

            # Prepare masks for op-wise gradient accumulation.
            # TODO: Record actual transitions (e.g. for model 1S and higher)
            # and repeat those here
            mask = transitions_t_f
            masks = [mask, mask.dimshuffle(0, "x"),
                     mask.dimshuffle(0, "x", "x")]

            # Accumulate gradients from the embedding projection network into
            # the stack op gradients.
            if f_proj_delta is not None:
                # Look up raw buffer top for this timestep -- i.e., buffer top
                # *before* the op at this timestep was performed. This was the
                # input to the projection network at this timestep.
                proj_input = cuda.AdvancedSubtensor1Floats("B_raw_buffer_top")(
                    self._raw_buffer_t, buffer_cur_t + self._buffer_shift)

                proj_inputs = (proj_input,)
                if self.use_input_dropout:
                    embedding_dropout_mask = cuda.AdvancedSubtensor1Floats("B_buffer_dropout")(
                        self._embedding_dropout_masks, buffer_cur_t + self._buffer_shift)
                    proj_inputs = (proj_input, embedding_dropout_mask)

                # Compute separate graphs based on gradient from above.
                # NB: We discard the delta_inp return here. The delta_inp
                # should actually be passed back to the raw embedding
                # parameters, but we don't have any reason to support this in
                # practice. (Either we backprop to embeddings or project them
                # and learn the projection -- not both.)
                if r_delta_inp[2] is not None:
                    _, m_proj_delta_wrt = f_proj_delta(proj_inputs,
                                                       (r_delta_inp[2],))
                    r_delta_wrt = util.merge_update_lists(r_delta_wrt, m_proj_delta_wrt)

                # If we shifted (moved the buffer top onto the stack), the
                # gradient from above is a combination of the accumulated stack
                # gradient (main_grad) and any buffer top deltas from the shift
                # function (e.g. tracking LSTM gradient).
                embedding_grad = main_grad
                if s_delta_inp[2] is not None:
                    embedding_grad += s_delta_inp[2]
                _, p_proj_delta_wrt = f_proj_delta(proj_inputs,
                                                   (embedding_grad,))
                s_delta_wrt = util.merge_update_lists(s_delta_wrt, p_proj_delta_wrt)

            ###########
            # STEP 1. #
            ###########
            # Accumulate deltas onto the graph inputs, switching over
            # shift/reduce decisions.
            stacks = (stack_bwd_next, stack_bwd_next,
                      (compute_embedding_gradients and dE) or None)
            cursors = (t_c1, t_c2,
                       (compute_embedding_gradients and buffer_ids_t) or None)
            # Handle potential aux bwd stacks.
            stacks += extra_bwd
            cursors += ((t_c1,)) * len(extra_bwd)
            new_stacks = {}
            for stack, cursor, r_delta, s_delta in zip(stacks, cursors, r_delta_inp, s_delta_inp):
                if stack is None or cursor is None:
                    continue
                elif r_delta is None and s_delta is None:
                    # Disconnected gradient.
                    continue

                base = new_stacks.get(stack, stack)
                mask_i = masks[(r_delta or s_delta).ndim - 1]
                if r_delta is None:
                    delta = (1. - mask_i) * s_delta
                elif s_delta is None:
                    delta = mask_i * r_delta
                else:
                    delta = mask_i * r_delta + (1. - mask_i) * s_delta

                # Run subtensor update on associated structure using the
                # current cursor.
                new_stack = cuda.AdvancedIncSubtensor1Floats(inplace=True)(
                    base, delta, cursor)
                new_stacks[stack] = new_stack

            ###########
            # STEP 2. #
            ###########
            # Accumulate deltas with respect to the variables in the wrt store,
            # again switching over shift/reduce decision.
            new_wrt_deltas = {}
            wrt_data = enumerate(zip(wrt, zero_jac_wrts, wrt_deltas,
                                     r_delta_wrt, s_delta_wrt))
            for i, (wrt_var, wrt_zero, accum_delta, r_delta, s_delta) in wrt_data:
                if r_delta is None and s_delta is None:
                    # Disconnected gradient.
                    continue

                # Check that tensors returned by delta functions match shape
                # expectations.
                assert r_delta is None or accum_delta.ndim == r_delta.ndim - 1, \
                    "%s %i %i" % (wrt_var.name, accum_delta.ndim, r_delta.ndim)
                assert s_delta is None or accum_delta.ndim == s_delta.ndim - 1, \
                    "%s %i %i" % (wrt_var.name, accum_delta.ndim, s_delta.ndim)

                mask_i = masks[(r_delta or s_delta).ndim - 1]
                if r_delta is None:
                    delta = T.switch(mask_i, wrt_zero, s_delta)
                elif s_delta is None:
                    delta = T.switch(mask_i, r_delta, wrt_zero)
                else:
                    delta = T.switch(mask_i, r_delta, s_delta)
                # TODO: Is this at all efficient? (Bring back GPURowSwitch?)
                delta = delta.sum(axis=0)
                # TODO: we want this to be inplace
                new_wrt_deltas[accum_delta] = accum_delta + delta

            # On shift ops, backprop the stack_bwd error onto the embedding
            # projection network / embedding parameters.
            # TODO make sparse?
            if compute_embedding_gradients:
                new_stacks[dE] = cuda.AdvancedIncSubtensor1Floats(inplace=True)(
                    new_stacks.get(dE, dE), (1. - masks[1]) * main_grad, buffer_ids_t)

            updates = dict(new_wrt_deltas.items() + new_stacks.items())
            updates = util.prepare_updates_dict(updates)

            return updates

        # TODO: These should come from forward pass -- not fixed -- in model
        # 1S, etc.
        transitions_f = T.cast(self.transitions.dimshuffle(1, 0),
                               dtype=theano.config.floatX)

        ts_f = T.cast(T.arange(1, self.seq_length + 1), dtype=theano.config.floatX)

        # Representation of buffer using embedding indices rather than values
        id_buffer = T.cast(self.X.flatten(), theano.config.floatX)
        # Build sequence of buffer pointers, where buf_ptrs[i] indicates the
        # buffer pointer values *before* computation at timestep *i* proceeds.
        # (This means we need to slice off the last actual buf_ptr output and
        # prepend a dummy.)
        buf_ptrs = T.concatenate([T.zeros((1, self.batch_size,)),
                                  self.buf_ptrs[:-1]], axis=0)

        sequences = [ts_f, transitions_f, self.stack_2_ptrs, buf_ptrs]
        outputs_info = []

        # Shared variables: Accumulated wrt deltas and bwd stacks.
        non_sequences = [dE] + wrt_vars
        non_sequences += [self.stack_bwd] + self.aux_bwd_stacks
        # More auxiliary data
        non_sequences += [id_buffer, self.final_stack]

        # More helpers (not referenced directly in code, but we need to include
        # them as non-sequences to satisfy scan strict mode)
        aux_data = [self.stack, self.buffer_t] + self.aux_stacks + self.final_aux_stacks
        aux_data += [self.X, self.transitions, self._raw_buffer_t]
        if self.use_input_dropout:
            aux_data.append(self._embedding_dropout_masks)
        aux_data += self._vs.vars.values() + extra_cost_inputs
        non_sequences += list(set(aux_data))

        bscan_ret, self.bscan_updates = theano.scan(
                step_b, sequences, outputs_info, non_sequences,
                go_backwards=True,
                n_steps=self.seq_length,
#                strict=True,
                name=self._prefix + "stack_bwd")

        self.gradients = {wrt_i: self.bscan_updates.get(wrt_var)
                          for wrt_i, wrt_var in zip(wrt, wrt_vars)}
        if compute_embedding_gradients:
            self.embedding_gradients = self.bscan_updates[dE]
