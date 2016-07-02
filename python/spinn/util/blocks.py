"""High-level modular Theano-based network components."""

from collections import OrderedDict
from functools import partial

import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from spinn.util import NUM_TRANSITION_TYPES


numpy_random = np.random.RandomState(1234)
theano_random = MRG_RandomStreams(numpy_random.randint(999999))


def UniformInitializer(range):
    return lambda shape, **kwargs: np.random.uniform(-range, range, shape)


def HeKaimingInitializer():
    def HeKaimingInit(shape, real_shape=None):
        # Calculate fan-in / fan-out using real shape if given as override
        fan = real_shape or shape

        return np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
                                size=shape)
    return HeKaimingInit


def NormalInitializer(std):
    return lambda shape, **kwargs: np.random.normal(0.0, std, shape)


def ZeroInitializer():
    return lambda shape, **kwargs: np.zeros(shape, dtype=theano.config.floatX)


def OneInitializer():
    return lambda shape, **kwargs: np.ones(shape, dtype=theano.config.floatX)


def TreeLSTMBiasInitializer():
    def init(shape):
        hidden_dim = shape[0] / 5
        value = np.zeros(shape)
        value[hidden_dim:3*hidden_dim] = 1
        return value
    return init


def LSTMBiasInitializer():
    def init(shape):
        hidden_dim = shape[0] / 4
        value = np.zeros(shape)
        value[hidden_dim:2*hidden_dim] = 1
        return value
    return init


def DoubleIdentityInitializer(range):
    def init(shape):
        half_d = shape[0] / 2
        double_identity = np.concatenate((
            np.identity(half_d), np.identity(half_d)))
        return double_identity + UniformInitializer(range)(shape)
    return init


def HeKaimingResidualLayerSet(inp, inp_dim, vs, training_mode, name="resnet_stack", dropout_keep_rate=1.0, depth=2):
    # From http://arxiv.org/pdf/1603.05027v2.pdf
    addin = inp
    for i in range(depth):
        addin = BatchNorm(addin, inp_dim, vs, name + "/" + str(i), training_mode)
        if dropout_keep_rate < 1.0:
            addin = Dropout(addin, dropout_keep_rate, training_mode) 
        addin = T.maximum(addin, 0) # ReLU
        addin = Linear(addin, inp_dim, inp_dim, vs, name=name + "/" + str(i))
    return inp + addin


def ReLULayer(inp, inp_dim, outp_dim, vs, name="relu_layer", use_bias=True, initializer=None):
    pre_nl = Linear(inp, inp_dim, outp_dim, vs, name, use_bias, initializer)
    # ReLU isn't present in this version of Theano.
    outp = T.maximum(pre_nl, 0)

    return outp


def Linear(inp, inp_dim, outp_dim, vs, name="linear_layer", use_bias=True, initializer=None):
    if isinstance(inp, tuple):
        assert isinstance(inp_dim, tuple)
        # Build initializers which are aware of the real shape of the overall
        # (unsplit) matrix.
        real_inp_dim = sum(inp_dim)
        initializer = partial(initializer or vs.default_initializer,
                              real_shape=(real_inp_dim, outp_dim))

        try:
            Ws = [vs.add_param("%s_W%i" % (name, i), (dim_i, outp_dim),
                               initializer=initializer)
                  for i, dim_i in enumerate(inp_dim)]
        except TypeError, e:
            raise RuntimeError(
                "TypeError in vs initialization for split Gemm. Does the "
                "initializer you provided (%s) support real_shape?"
                % initializer, e)

        outp = T.dot(inp[0], Ws[0])
        for inp_i, W_i in zip(inp[1:], Ws[1:]):
            # TODO inplace add?
            outp += T.dot(inp_i, W_i)
    else:
        W = vs.add_param("%s_W" %
                         name, (inp_dim, outp_dim), initializer=initializer)
        outp = inp.dot(W)

    if use_bias:
        b = vs.add_param("%s_b" % name, (outp_dim,),
                         initializer=ZeroInitializer())
        outp += b

    return outp


def Dropout(inp, keep_rate, apply_dropout, dropout_mask=None,
            return_mask=False):
    """Apply dropout to a set of activations.

    Args:
      inp: Input vector.
      keep_rate: Dropout parameter. 1.0 entails no dropout.
      apply_dropout: A Theano scalar indicating whether to apply dropout (1.0)
        or eval-mode rescaling (0.0).
    """
    # TODO(SB): Investigate whether a Theano conditional would be faster than the linear combination below.

    dropout_mask = (dropout_mask or
                    theano_random.binomial(n=1, p=keep_rate, size=inp.shape,
                                           dtype=theano.config.floatX))

    dropout_candidate = dropout_mask * inp
    rescaling_candidate = keep_rate * inp
    result = apply_dropout * dropout_candidate + (1 - apply_dropout) * rescaling_candidate

    if return_mask:
        return result, dropout_mask
    return result


def IdentityLayer(inp, inp_dim, outp_dim, vs, name="identity_layer", use_bias=True, initializer=None):
    """An identity function that takes the same parameters as the above layers."""
    assert inp_dim == outp_dim, "Identity layer requires inp_dim == outp_dim."
    return inp


def TreeLSTMLayer(lstm_prev, external_state, full_memory_dim, vs, name="tree_lstm", initializer=None, external_state_dim=0):
    assert full_memory_dim % 2 == 0, "Input is concatenated (h, c); dim must be even."
    hidden_dim = full_memory_dim / 2

    assert isinstance(lstm_prev, tuple)
    l_prev, r_prev = lstm_prev

    real_shape = (hidden_dim * 2 + external_state_dim, hidden_dim * 5)
    initializer_children = partial(initializer or vs.default_initializer,
                                   real_shape=real_shape)
    W_l = vs.add_param("%s/W_l" % name, (hidden_dim, hidden_dim * 5),
                       initializer=initializer_children)
    W_r = vs.add_param("%s/W_r" % name, (hidden_dim, hidden_dim * 5),
                       initializer=initializer_children)
    if external_state_dim > 0:
        W_ext = vs.add_param("%s/W_ext" % name, (external_state_dim, hidden_dim * 5),
                             initializer=initializer_children)
    b = vs.add_param("%s/b" % name, (hidden_dim * 5,),
                     initializer=TreeLSTMBiasInitializer())

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Decompose previous LSTM value into hidden and cell value
    l_h_prev = l_prev[:, :hidden_dim]
    l_c_prev = l_prev[:,  hidden_dim:]
    r_h_prev = r_prev[:, :hidden_dim]
    r_c_prev = r_prev[:,  hidden_dim:]

    gates = T.dot(l_h_prev, W_l) + T.dot(r_h_prev, W_r) + b
    if external_state_dim > 0:
        gates += T.dot(external_state, W_ext)

    # Compute and slice gate values
    i_gate, fl_gate, fr_gate, o_gate, cell_inp = [slice_gate(gates, i) for i in range(5)]

    # Apply nonlinearities
    i_gate = T.nnet.sigmoid(i_gate)
    fl_gate = T.nnet.sigmoid(fl_gate)
    fr_gate = T.nnet.sigmoid(fr_gate)
    o_gate = T.nnet.sigmoid(o_gate)
    cell_inp = T.tanh(cell_inp)

    # Compute new cell and hidden value
    c_t = fl_gate * l_c_prev + fr_gate * r_c_prev + i_gate * cell_inp
    h_t = o_gate * T.tanh(c_t)

    return T.concatenate([h_t, c_t], axis=1)


def LSTMLayer(lstm_prev, inp, inp_dim, full_memory_dim, vs, name="lstm", initializer=None):
    assert full_memory_dim % 2 == 0, "Input is concatenated (h, c); dim must be even."
    hidden_dim = full_memory_dim / 2

    # gate biases
    # TODO(jgauthier): support excluding params from regularization
    b = vs.add_param("%s_b" % name, (hidden_dim * 4,),
                     initializer=LSTMBiasInitializer())

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Decompose previous LSTM value into hidden and cell value
    h_prev = lstm_prev[:, :hidden_dim]
    c_prev = lstm_prev[:, hidden_dim:]

    # Compute and slice gate values
    # input -> hidden mapping
    gates = Linear(inp, inp_dim, hidden_dim * 4, vs,
                   name="%s/inp/linear" % name,
                   initializer=initializer, use_bias=False)
    # hidden -> hidden mapping
    gates += Linear(h_prev, hidden_dim, hidden_dim * 4, vs,
                    name="%s/hid/linear" % name,
                    initializer=initializer, use_bias=False)
    gates += b
    i_gate, f_gate, o_gate, cell_inp = [slice_gate(gates, i) for i in range(4)]

    # Apply nonlinearities
    i_gate = T.nnet.sigmoid(i_gate)
    f_gate = T.nnet.sigmoid(f_gate)
    o_gate = T.nnet.sigmoid(o_gate)
    cell_inp = T.tanh(cell_inp)

    # Compute new cell and hidden value
    c_t = f_gate * c_prev + i_gate * cell_inp
    h_t = o_gate * T.tanh(c_t)

    return T.concatenate([h_t, c_t], axis=1)


def MLP(inp, inp_dim, outp_dim, vs, layer=ReLULayer, hidden_dims=None,
        name="mlp", initializer=None):
    if hidden_dims is None:
        hidden_dims = []

    prev_val = inp
    dims = [inp_dim] + hidden_dims + [outp_dim]
    for i, (src_dim, tgt_dim) in enumerate(zip(dims, dims[1:])):
        prev_val = layer(prev_val, src_dim, tgt_dim, vs,
                         use_bias=True,
                         name="%s/%i" % (name, i),
                         initializer=initializer)
    return prev_val


def SGD(cost, params, lr=0.01):
    grads = T.grad(cost, params)

    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        new_values[param] = param - lr * grad

    return new_values


def EmbeddingSGD(cost, embedding_matrix, lr=0.01, used_embeddings=None):
    new_values = OrderedDict()

    if used_embeddings:
        grads = T.grad(cost, wrt=used_embeddings)
        new_value = (used_embeddings,
                     T.inc_subtensor(used_embeddings, -lr * grads))
    else:
        new_values = SGD(cost, [embedding_matrix], lr)
        new_value = (embedding_matrix, new_values[embedding_matrix])

    return new_value


def Momentum(cost, params, lr=0.01, momentum=0.9):
    grads = T.grad(cost, params)

    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        param_val = param.get_value(borrow=True)
        # momentum value
        m = theano.shared(np.zeros(param_val.shape, dtype=param_val.dtype))
        # compute velocity
        v = lr * grad + momentum * m

        new_values[m] = v
        new_values[param] = param - v

    return new_values


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6, grads=None):
    # From:
    # https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py
    if grads is None:
        grads = T.grad(cost=cost, wrt=params)
    assert len(grads) == len(params)

    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(np.zeros_like(p.get_value(), dtype=np.float32),
                            name="%s/rms/acc" % p.name)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def BatchNorm(x, input_dim, vs, name, training_mode, axes=[0], momentum=0.9):
    """Apply simple batch normalization.
    This requires introducing new learned scale parameters, so it's
    important to use unique names unless you're sure you want to share
    these parameters.
    """

    # Create the trained gamma and beta parameters.
    g = vs.add_param("%s_bn_g" % name, (input_dim),
        initializer=OneInitializer())
    b = vs.add_param("%s_bn_b" % name, (input_dim),
        initializer=ZeroInitializer())

    # Create the training set moving averages for test time use.
    tracking_std = vs.add_param("%s_bn_ts" % name, (input_dim),
        initializer=OneInitializer(),
        trainable=False)
    tracking_mean = vs.add_param("%s_bn_tm" % name, (input_dim),
        initializer=ZeroInitializer(),
        trainable=False)

    # Compute the empirical mean and std.
    mean = x.mean(axis=axes, keepdims=True)
    std = T.sqrt(x.var(axis=axes, keepdims=True) + 1e-12)

    # Update the moving averages.
    vs.add_nongradient_update(tracking_std, (momentum * tracking_std + (1 - momentum) * std).flatten(ndim=1))
    vs.add_nongradient_update(tracking_mean, (momentum * tracking_mean + (1 - momentum) * mean).flatten(ndim=1))

    # Switch between train and test modes.
    effective_mean = mean * training_mode + tracking_mean * (1 - training_mode)
    effective_std = std * training_mode + tracking_std * (1 - training_mode)

    # Apply batch norm.
    return (x - effective_mean) * (g / effective_std) + b


def TrackingUnit(state_prev, inp, inp_dim, hidden_dim, vs,
                 make_logits=True, logits_use_cell=False, name="track_unit"):
    """
    Defines a basic recurrent tracking unit that optionally predicts actions.

    This is just an LSTM layer which combines given state and input, and
    optionally uses the resulting state to predict parser actions.

    Arguments:
        state_prev: Theano batch of previous tracking state outputs
        inp: Theano batch of inputs to tracking unit
        inp_dim: Dimensionality of `inp`
        hidden_dim: Size of complete LSTM state representation in `state_prev`
            (both cell and hidden values)
        vs: VariableStore
        make_logits: If true, also compute an output layer from the generated
            state representation.
        logits_use_cell: When producing logits, use both the LSTM hidden value
            and cell value. If false, just use the hidden value. Only has an
            effect when `make_logits` is `True`.
        name:

    Returns:
        state: Theano batch of state representations for this timestep; same
            size as `state_prev`
        logits: If `make_logits` is `True`, a Theano output layer batch of
            dimension `batch_size * NUM_TRANSITION_TYPES`. Otherwise do not
            use.
    """
    # Pass previous state and input to an LSTM layer.
    state = LSTMLayer(state_prev, inp, inp_dim, 2 * hidden_dim, vs, name="%s/lstm" % name)

    if make_logits:
        if logits_use_cell:
            pred_inp = state
            pred_inp_dim = hidden_dim * 2
        else:
            pred_inp = state[:, :hidden_dim]
            pred_inp_dim = hidden_dim

        # Pass LSTM states through a Linear layer to predict the next transition.
        logits = Linear(pred_inp, pred_inp_dim, NUM_TRANSITION_TYPES, vs, name="%s/linear" % name)
    else:
        logits = 0.0

    return state, logits


def TreeWangJiangAttentionUnit(attention_state_prev_l, attention_state_prev_r, current_stack_top,
        premise_stack_tops, projected_stack_tops, attention_dim, vs, name="attention_unit",
        initializer=None):
    """
    This is for use in a Wang and Jiang style mLSTM attention formalism where a TreeLSTM, rather than
    an LSTM RNN, accumulates attention states. In this setting, the model should contain a single step
    of 2 * model_dim dimensions, where the left [0:model_dim] half contains the TreeLSTM composition states
    and the right [model_dim:2 * model_dim] half contains the mTreeLSTM states.

    Args:
      attention_state_prev_{l, r}: The attention results for the children of this node, if present.
        Else, zero vectors.
      current_stack_top: The current stack top (h state only, if applicable).
      premise_stack_tops: The values to do attention over.
      projected_stack_tops: Projected vectors to use to produce an attentive
          weighting alpha_t.
      attention_dim: The dimension of the vectors over which to do attention.
      vs: A variable store for the learned parameters.
      name: An identifier for the learned parameters in this unit.
      initializer: Used to initialize the learned parameters.

    Dimension notation:
      B : Batch size
      k : Model dim
      L : num_transitions
    """
    W_h = vs.add_param("%s_W_h" % name, (attention_dim, attention_dim), initializer=initializer)
    W_rl = vs.add_param("%s_W_rl" % name, (attention_dim, attention_dim), initializer=initializer)
    W_rr = vs.add_param("%s_W_rr" % name, (attention_dim, attention_dim), initializer=initializer)
    w = vs.add_param("%s_w" % name, (attention_dim,), initializer=initializer)

    W_h__h_t = T.dot(current_stack_top, W_h)
    W_rl__l_prev = T.dot(attention_state_prev_l[:,:attention_dim], W_rl)
    W_rr__r_prev = T.dot(attention_state_prev_r[:,:attention_dim], W_rr)

    # Shape: L x B x k
    M_t = T.tanh(projected_stack_tops + (W_h__h_t +  W_rl__l_prev + W_rr__r_prev))

    # Shape: B x L
    alpha_t = T.nnet.softmax(T.dot(M_t, w).T)

    # Shape B x k
    Y__alpha_t = T.sum(premise_stack_tops * alpha_t.T[:, :, np.newaxis], axis=0)

    mlstm_input = T.concatenate([Y__alpha_t, current_stack_top], axis=1)
    lstm_prev = (attention_state_prev_l, attention_state_prev_r)
    r_t = TreeLSTMLayer(lstm_prev, mlstm_input, 2 * attention_dim, vs,
            name="%s/lstm" % name, external_state_dim=2 * attention_dim)

    return r_t


def WangJiangAttentionUnit(attention_state_prev, current_stack_top, premise_stack_tops, projected_stack_tops, attention_dim,
                    vs, name="attention_unit", initializer=None):
    """
    Args:
      attention_state_prev: The output of this unit at the previous time step.
      current_stack_top: The current stack top (h state only, if applicable).
      premise_stack_tops: The values to do attention over.
      projected_stack_tops: Projected vectors to use to produce an attentive
          weighting alpha_t.
      attention_dim: The dimension of the vectors over which to do attention.
      vs: A variable store for the learned parameters.
      name: An identifier for the learned parameters in this unit.
      initializer: Used to initialize the learned parameters.

    Dimension notation:
      B : Batch size
      k : Model dim
      L : num_transitions
    """
    W_h = vs.add_param("%s_W_h" % name, (attention_dim, attention_dim), initializer=initializer)
    W_r = vs.add_param("%s_W_r" % name, (attention_dim, attention_dim), initializer=initializer)
    w = vs.add_param("%s_w" % name, (attention_dim,), initializer=initializer)

    W_h__h_t = T.dot(current_stack_top, W_h)
    W_r__r_t_prev = T.dot(attention_state_prev[:,:attention_dim], W_r)

    # Shape: L x B x k
    M_t = T.tanh(projected_stack_tops + (W_h__h_t + W_r__r_t_prev))

    # Shape: B x L
    alpha_t = T.nnet.softmax(T.dot(M_t, w).T)

    # Shape B x k
    Y__alpha_t = T.sum(premise_stack_tops * alpha_t.T[:, :, np.newaxis], axis=0)

    mlstm_input = T.concatenate([Y__alpha_t, current_stack_top], axis=1)

    r_t = LSTMLayer(attention_state_prev, mlstm_input, 2 * attention_dim, 2 * attention_dim, vs, name="%s/lstm" % name)

    return r_t

def TreeThangAttentionUnit(attention_state_prev_l, attention_state_prev_r, current_stack_top,
        premise_stack_tops, projected_stack_tops, attention_dim, vs, name="attention_unit",
        initializer=None):
    """
    Args:
      attention_state_prev_{l, r}: The attention results for the children of this node, if present.
        Else, zero vectors.
      current_stack_top: The current stack top (h state only, if applicable).
      premise_stack_tops: The values to do attention over.
      projected_stack_tops: Projected vectors to use to produce an attentive
          weighting alpha_t.
      attention_dim: The dimension of the vectors over which to do attention.
      vs: A variable store for the learned parameters.
      name: An identifier for the learned parameters in this unit.
      initializer: Used to initialize the learned parameters.

    Dimension notation:
      B : Batch size
      k : Model dim
      L : num_transitions
    """
    # Shape: B x L
    score = T.sum(projected_stack_tops * current_stack_top, axis=2).T
    alpha_t = T.nnet.softmax(score)

    # Shape B x k
    Y__alpha_t = T.sum(premise_stack_tops * alpha_t.T[:, :, np.newaxis], axis=0)

    mlstm_input = T.concatenate([Y__alpha_t, current_stack_top], axis=1)
    lstm_prev = (attention_state_prev_l, attention_state_prev_r)
    r_t = TreeLSTMLayer(lstm_prev, mlstm_input, 2 * attention_dim, vs,
            name="%s/lstm" % name, external_state_dim=2 * attention_dim)

    return r_t


def ThangAttentionUnit(attention_state_prev, current_stack_top, premise_stack_tops, projected_stack_tops, attention_dim,
                    vs, name="attention_unit", initializer=None):
    """
    Args:
      attention_state_prev: The output of this unit at the previous time step.
      current_stack_top: The current stack top (h state only, if applicable).
      premise_stack_tops: The values to do attention over.
      projected_stack_tops: Projected vectors to use to produce an attentive
          weighting alpha_t.
      attention_dim: The dimension of the vectors over which to do attention.
      vs: A variable store for the learned parameters.
      name: An identifier for the learned parameters in this unit.
      initializer: Used to initialize the learned parameters.

    Dimension notation:
      B : Batch size
      k : Model dim
      L : num_transitions
    """
    # Shape: B x L
    score = T.sum(projected_stack_tops * current_stack_top, axis=2).T
    alpha_t = T.nnet.softmax(score)

    # Shape B x k
    Y__alpha_t = T.sum(premise_stack_tops * alpha_t.T[:, :, np.newaxis], axis=0)

    mlstm_input = T.concatenate([Y__alpha_t, current_stack_top], axis=1)

    r_t = LSTMLayer(attention_state_prev, mlstm_input, 2 * attention_dim, 2 * attention_dim, vs, name="%s/lstm" % name)

    return r_t

def RocktaschelAttentionUnit(attention_state_prev, current_stack_top, premise_stack_tops, projected_stack_tops, attention_dim,
                    vs, name="attention_unit", initializer=None):
    """
    Args:
      attention_state_prev: The output of this unit at the previous time step.
      current_stack_top: The current stack top (h state only, if applicable).
      premise_stack_tops: The values to retrieve using attention.
      projected_stack_tops: Projected vectors to use to produce an attentive
          weighting alpha_t.
      attention_dim: The dimension of the vectors over which to do attention.
      vs: A variable store for the learned parameters.
      name: An identifier for the learned parameters in this unit.
      initializer: Used to initialize the learned parameters.

    Dimension notation:
      B : Batch size
      k : Model dim
      L : num_transitions
    """

    W_h = vs.add_param("%s_W_h" % name, (attention_dim, attention_dim), initializer=initializer)
    W_r = vs.add_param("%s_W_r" % name, (attention_dim, attention_dim), initializer=initializer)
    W_t = vs.add_param("%s_W_t" % name, (attention_dim, attention_dim), initializer=initializer)
    w = vs.add_param("%s_w" % name, (attention_dim,), initializer=initializer)

    W_h__h_t = T.dot(current_stack_top, W_h)
    W_r__r_t_prev = T.dot(attention_state_prev, W_r)

    # Vector-by-matrix addition here: (Right?)
    # Shape: L x B x k
    M_t = T.tanh(projected_stack_tops + (W_h__h_t + W_r__r_t_prev))

    # Shape: B x L
    alpha_t = T.nnet.softmax(T.dot(M_t, w).T)

    # Shape B x k
    Y__alpha_t = T.sum(premise_stack_tops * alpha_t.T[:, :, np.newaxis], axis=0)

    # Mysterious Rocktaschel-style RNN update step.
    r_t = Y__alpha_t + T.tanh(T.dot(attention_state_prev, W_t))
    return r_t


def AttentionUnitFinalRepresentation(final_attention_state, final_stack_top, attention_dim, vs, initializer=None, name="attention_unit_final"):
    """Produces the complete representation of the aligned sentence pair."""

    W_p = vs.add_param("%s_W_p" % name, (attention_dim, attention_dim), initializer=initializer)
    W_x = vs.add_param("%s_W_x" % name, (attention_dim, attention_dim), initializer=initializer)
    h_final = T.tanh(T.dot(final_attention_state, W_p) + T.dot(final_stack_top, W_x))
    return h_final


def AttentionUnitInit(premise_stack_tops, attention_dim, vs, initializer=None, name="attention_unit_init"):
    """Does an initial reweighting on the input vectors that will be used for attention.

    Unlike the units above, this only needs to be called once per batch, not at every step."""

    W_y = vs.add_param("%s_W_y" % name, (attention_dim, attention_dim), initializer=initializer)
    return T.dot(premise_stack_tops, W_y)


