from collections import namedtuple, OrderedDict
import cPickle
from functools import partial, wraps
import itertools
import math
import random

import numpy as np
import theano
from theano import ifelse, tensor as T
from theano.compile.sharedvalue import SharedVariable
from theano.gof.fg import MissingInputError
from theano.sandbox.cuda import HostFromGpu
from theano.sandbox.rng_mrg import MRG_RandomStreams

from spinn import cuda_util

numpy_random = np.random.RandomState(1234)
theano_random = MRG_RandomStreams(numpy_random.randint(999999))

ModelSpec_ = namedtuple("ModelSpec", ["model_dim", "word_embedding_dim",
                                      "batch_size", "vocab_size", "seq_length",
                                      "model_visible_dim"])
def ModelSpec(*args, **kwargs):
    args = dict(zip(ModelSpec_._fields, args))
    args.update(kwargs)

    # Defaults
    if "model_visible_dim" not in args:
        args["model_visible_dim"] = args["model_dim"]

    return ModelSpec_(**args)

# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "*PADDING*"

# Temporary hack: Map UNK to "_" when loading pretrained embedding matrices:
# it's a common token that is pretrained, but shouldn't look like any content words.
UNK_TOKEN = "_"

CORE_VOCABULARY = {PADDING_TOKEN: 0,
                   UNK_TOKEN: 1}

# Allowed number of transition types : currently PUSH : 0 and MERGE : 1
NUM_TRANSITION_TYPES = 2


def UniformInitializer(range):
    return lambda shape, **kwargs: np.random.uniform(-range, range, shape)


def HeKaimingInitializer():
    def HeKaimingInit(shape, real_shape=None):
        # Calculate fan-in / fan-out using real shape if given as override
        fan = real_shape or shape

        return np.random.normal(scale=math.sqrt(4.0/(fan[0] + fan[1])),
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


class VariableStore(object):

    def __init__(self, prefix="vs", default_initializer=HeKaimingInitializer(), logger=None):
        self.prefix = prefix
        self.default_initializer = default_initializer
        self.vars = OrderedDict()  # Order is used in saving and loading
        self.savable_vars = OrderedDict()
        self.trainable_vars = OrderedDict()
        self.logger = logger
        self.nongradient_updates = OrderedDict()

    def add_param(self, name, shape, initializer=None, savable=True, trainable=True):
        if not initializer:
            initializer = self.default_initializer

        if name not in self.vars:
            full_name = "%s/%s" % (self.prefix, name)
            if self.logger:
                self.logger.Log(
                    "Created variable " + full_name + " shape: " + str(shape), level=self.logger.DEBUG)
            init_value = initializer(shape).astype(theano.config.floatX)
            self.vars[name] = theano.shared(init_value,
                                            name=full_name)
            if savable:
                self.savable_vars[name] = self.vars[name]
            if trainable:
                self.trainable_vars[name] = self.vars[name]

        return self.vars[name]

    def save_checkpoint(self, filename="vs_ckpt", keys=None, extra_vars=[]):
        if not keys:
            keys = self.savable_vars
        save_file = open(filename, 'w')  # this will overwrite current contents
        for key in keys:
            cPickle.dump(self.vars[key].get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
        for var in extra_vars:
            cPickle.dump(var, save_file, -1)
        save_file.close()

    def load_checkpoint(self, filename="vs_ckpt", keys=None, num_extra_vars=0, skip_saved_unsavables=False):
        if skip_saved_unsavables:
            keys = self.vars
        else:
            if not keys:
                keys = self.savable_vars
        save_file = open(filename)
        for key in keys:
            if skip_saved_unsavables and key not in self.savable_vars:
                if self.logger:
                    full_name = "%s/%s" % (self.prefix, key)
                    self.logger.Log(
                        "Not restoring variable " + full_name, level=self.logger.DEBUG)
                _ = cPickle.load(save_file) # Discard
            else:
                if self.logger:
                    full_name = "%s/%s" % (self.prefix, key)
                    self.logger.Log(
                        "Restoring variable " + full_name, level=self.logger.DEBUG)
            self.vars[key].set_value(cPickle.load(save_file), borrow=True)

        extra_vars = []
        for _ in range(num_extra_vars):
            extra_vars.append(cPickle.load(save_file))
        return extra_vars

    def add_nongradient_update(self, variable, new_value):
        # Track an update that should be applied during training but that aren't gradients.
        # self.nongradient_updates should be fed as an update to theano.function().
        self.nongradient_updates[variable] = new_value


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


def TrackingUnit(state_prev, inp, inp_dim, hidden_dim, vs, name="track_unit", make_logits=True):
    # Pass previous state and input to an LSTM layer.
    state = LSTMLayer(state_prev, inp, inp_dim, 2 * hidden_dim, vs, name="%s/lstm" % name)

    if make_logits:
        # Pass LSTM states through a Linear layer to predict the next transition.
        logits = Linear(state, 2 * hidden_dim, NUM_TRANSITION_TYPES, vs, name="%s/linear" % name)
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


def tensorx(name, ndim, dtype=theano.config.floatX):
    return T.TensorType(dtype, (False,) * ndim)(name)


def zeros_nobroadcast(shape, dtype=theano.config.floatX):
    zeros = T.zeros(shape, dtype=dtype)
    zeros = T.unbroadcast(zeros, *range(len(shape)))
    return zeros


def merge_update_lists(xs, ys):
    """
    Merge two update lists:

    - adding where `xs[i] is not None and ys[i] is not None`
    - copying `xs[i]` if `xs[i] is not None`
    - copying `ys[i]` otherwise
    """

    assert len(xs) == len(ys), "%i %i" % (len(xs), len(ys))
    ret = []

    for x, y in zip(xs, ys):
        if y is None:
            ret.append(x)
        elif x is None:
            ret.append(y)
        else:
            # Merge.
            ret.append(x + y)

    return ret


def batch_subgraph_gradients(g_in, wrt, f_g_out, batch_size=None,
                             extra_scan_inputs=None,
                             wrt_jacobian=True,
                             name="batch_subgraph_grad"):
    """
    Build gradients w.r.t. some cost on a subgraph of a larger graph.

    Let G be a feedforward subgraph for which we want to compute gradients.
    G has N_I inputs and N_O outputs.

    This function will compute the *unaccumulated* batch gradients on the
    subgraph G. That is, for a batch of M inputs, it allocates M separate
    gradient entries and returns these without aggregating them. This is useful
    for downstream uses that need to input-wise mask the gradients.

    Args:
        g_in: List of N_I inputs to G. Each element may be either a
            symbolic Theano input variable or an integer (signifying the number
            of dimensions of the input).
        wrt: Any variables inside G for which we should also collect gradients.
        f_g_out: Function which accepts N_I Theano vars and returns N_O Theano
            vars.

    Returns:
        A function which accepts two arguments, `b_in` and `b_grad`.

        `b_in` must be a list of N_I Theano batch variables representing inputs
        to the subgraph G. (Each element of `b_in` has a leading batch axis and
        is thus one dimension larger than its corresponding element of `g_in`).

        `b_grad` must be a list of N_O Theano batch variables representing
        cost gradients w.r.t. each of the graph outputs. Again, each element of
        the list has a leading batch axis and is thus one dimension larger than
        its corresponding output from `f_g_out`.

        The function returns `(d_in, d_wrt)`, where

        - `d_in` is a list of batch cost gradients with respect to each of the
          corresponding elements of `g_in`. Each element of `d_in` has a
          leading batch axis, and is thus one dimension larger than its
          corresponding `g_in` element.
        - `d_wrt` is a list of batch cost gradients with respect to each of the
          corresponding elements of `wrt`. Each element of `d_wrt` has a
          leading batch axis, and is thus one dimension larger than its
          corresponding `wrt` element.
    """

    wrt = tuple(wrt)

    def deltas(b_inps, b_grads):
        b_inps = tuple(b_inps)
        assert len(g_in) == len(b_inps), "%i %i" % (len(g_in), len(b_inps))

        # Build feedforward graph.
        b_out = f_g_out(*b_inps)
        # Make sure it's a list of outputs.
        b_out = [b_out] if not isinstance(b_out, (list, tuple)) else b_out

        def dot_grad_override(op, inp, grads):
            x, y = inp
            xdim, ydim = x.type.ndim, y.type.ndim

            # HACK: Get super grads
            gz, = grads
            xgrad, ygrad = op.grad(inp, grads)

            if xdim == ydim == 2:
                # HACK: Compute the Jacobian of this `dot` op. We will yield a
                # rank-3 tensor rather than a gradient matrix.
                ygrad = T.batched_dot(x.dimshuffle(0, 1, "x"),
                                    gz.dimshuffle(0, "x", 1))

            # TODO patternbroadcast?

            return xgrad, ygrad

        # Overrides which turn our "grad" call into a "jacobian" call!
        overrides = None
        if wrt_jacobian:
            overrides = {T.Dot: dot_grad_override}

        # Compute gradients of subgraph beginning at `g_in` and ending at `g_out`,
        # where the cost gradient w.r.t. each `g_out` is given by the corresponding
        # entry in `grads_above`.
        known_grads = dict(zip(b_out, b_grads))
        d_all = T.grad(cost=None, wrt=b_inps + wrt,
                       known_grads=known_grads,
                       consider_constant=b_inps,
                       disconnected_inputs="ignore",
                       return_disconnected="None",
                       use_overrides=set(wrt),
                       grad_overrides=overrides)
        d_in, d_wrt = d_all[:len(b_inps)], d_all[len(b_inps):]

        # Strip any GPU<->host transfers that might have crept into this
        # automatically constructed graph.
        d_wrt = map(cuda_util.strip_transfer, d_wrt)
        d_in = map(cuda_util.strip_transfer, d_in)
        if d_wrt:
            for i in range(len(d_wrt)):
                if d_wrt[i] is None:
                    continue
                # HACK: Strip off DimShuffle(Elemwise(DimShuffle(Sum))). This is what
                # comes out for bias gradients.. don't ask me why.
                if isinstance(d_wrt[i].owner.op, T.DimShuffle):
                    base = d_wrt[i].owner
                    if isinstance(base.inputs[0].owner.op, T.Elemwise):
                        base = base.inputs[0].owner
                        if isinstance(base.inputs[0].owner.op, T.DimShuffle):
                            base = base.inputs[0].owner
                            if isinstance(base.inputs[0].owner.op, T.Sum):
                                base = base.inputs[0].owner
                                d_wrt[i] = base.inputs[0]

        return d_in, d_wrt

    return deltas


def ensure_2d_arguments(f, squeeze_ret=True):
    """Decorator which ensures all of its function's arguments are 2D."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, T.TensorVariable):
                if arg.ndim == 1:
                    arg = arg.dimshuffle("x", 0)
                elif arg.ndim > 2:
                    raise RuntimeError("ensure_2d_arguments wrapped a function"
                                       " which received an %i-d argument. "
                                       "Don't know what to do.")
            new_args.append(arg)

        ret = f(*new_args, **kwargs)
        if squeeze_ret:
            if isinstance(ret, (list, tuple)):
                ret = [ret_i.squeeze() for ret_i in ret]
            elif isinstance(ret, T.TensorVariable):
                ret = ret.squeeze()
        return ret
    return wrapped


def prepare_updates_dict(updates):
    """
    Prepare a Theano `updates` dictionary.

    Ensure that both keys and values are valid entries.
    NB, this function is heavily coupled with its clients, and not intended for
    general use..
    """

    def prepare_key(key, val):
        if not isinstance(key, SharedVariable):
            if isinstance(key.owner.inputs[0], SharedVariable):
                # Extract shared from Update(shared)
                return key.owner.inputs[0]
            elif key.owner.inputs[0].owner.op.__class__ is HostFromGpu:
                if isinstance(key.owner.inputs[0].owner.inputs[0], SharedVariable):
                    # Extract shared from Update(HostFromGpu(shared))
                    return key.owner.inputs[0].owner.inputs[0]
            elif key.owner.op.__class__ is ifelse.IfElse:
                # Assume that 'true' condition of ifelse involves the intended
                # shared variable.
                return prepare_key(key.owner.inputs[1], val)

            raise ValueError("Invalid updates dict key/value: %s / %s"
                             % (key, val))
        return key

    return {prepare_key(key, val): val for key, val in updates.iteritems()}


def merge_updates(*updates_dicts):
    all_updates = OrderedDict()
    for updates_dict in updates_dicts:
        for k, v in updates_dict.iteritems():
            if k in all_updates:
                all_updates[k] += v
            else:
                all_updates[k] = v

    return all_updates


def TrimDataset(dataset, seq_length, eval_mode=False, sentence_pair_data=False):
    """Avoid using excessively long training examples."""
    if eval_mode:
        return dataset
    else:
        if sentence_pair_data:
            new_dataset = [example for example in dataset if
                len(example["premise_transitions"]) <= seq_length and
                len(example["hypothesis_transitions"]) <= seq_length]
        else:
            new_dataset = [example for example in dataset if len(
                example["transitions"]) <= seq_length]
        return new_dataset


def TokensToIDs(vocabulary, dataset, sentence_pair_data=False):
    """Replace strings in original boolean dataset with token IDs."""
    if sentence_pair_data:
        keys = ["premise_tokens", "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for key in keys:
        if UNK_TOKEN in vocabulary:
            unk_id = vocabulary[UNK_TOKEN]
            for example in dataset:
                example[key] = [vocabulary.get(token, unk_id)
                                     for token in example[key]]
        else:
            for example in dataset:
                example[key] = [vocabulary[token]
                                for token in example[key]]
    return dataset


def CropAndPadExample(example, left_padding, target_length, key, logger=None):
    """
    Crop/pad a sequence value of the given dict `example`.
    """
    if left_padding < 0:
        # Crop, then pad normally.
        # TODO: Track how many sentences are cropped, but don't log a message
        # for every single one.
        example[key] = example[key][-left_padding:]
        left_padding = 0
    right_padding = target_length - (left_padding + len(example[key]))
    example[key] = ([0] * left_padding) + \
        example[key] + ([0] * right_padding)


def CropAndPad(dataset, length, logger=None, sentence_pair_data=False):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    # Always make sure that the transitions are aligned at the left edge, so
    # the final stack top is the root of the tree. If cropping is used, it should
    # just introduce empty nodes into the tree.
    if sentence_pair_data:
        keys = [("premise_transitions", "num_premise_transitions", "premise_tokens"),
                ("hypothesis_transitions", "num_hypothesis_transitions", "hypothesis_tokens")]
    else:
        keys = [("transitions", "num_transitions", "tokens")]

    for example in dataset:
        for (transitions_key, num_transitions_key, tokens_key) in keys:
            example[num_transitions_key] = len(example[transitions_key])
            transitions_left_padding = length - example[num_transitions_key]
            shifts_before_crop_and_pad = example[transitions_key].count(0)
            CropAndPadExample(
                example, transitions_left_padding, length, transitions_key, logger=logger)
            shifts_after_crop_and_pad = example[transitions_key].count(0)
            tokens_left_padding = shifts_after_crop_and_pad - \
                shifts_before_crop_and_pad
            CropAndPadExample(
                example, tokens_left_padding, length, tokens_key, logger=logger)
    return dataset

def CropAndPadForRNN(dataset, length, logger=None, sentence_pair_data=False):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    if sentence_pair_data:
        keys = ["premise_tokens",
                "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for example in dataset:
        for tokens_key in keys:
            num_tokens = len(example[tokens_key])
            tokens_left_padding = length - num_tokens
            CropAndPadExample(
                example, tokens_left_padding, length, tokens_key, logger=logger)
    return dataset


def MakeTrainingIterator(sources, batch_size):
    # Make an iterator that exposes a dataset as random minibatches.

    def data_iter():
        dataset_size = len(sources[0])
        start = -1 * batch_size
        order = range(dataset_size)
        random.shuffle(order)

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            yield tuple(source[batch_indices] for source in sources)
    return data_iter()


def MakeEvalIterator(sources, batch_size):
    # Make a list of minibatches from a dataset to use as an iterator.
    # TODO(SB): Pad out the last few examples in the eval set if they don't
    # form a batch.

    print "WARNING: May be discarding eval examples."

    dataset_size = len(sources[0])
    data_iter = []
    start = -batch_size
    while True:
        start += batch_size

        if start >= dataset_size:
            break

        candidate_batch = tuple(source[start:start + batch_size]
                               for source in sources)

        if len(candidate_batch[0]) == batch_size:
            data_iter.append(candidate_batch)
        else:
            print "Skipping " + str(len(candidate_batch[0])) + " examples."
    return data_iter


def PreprocessDataset(dataset, vocabulary, seq_length, data_manager, eval_mode=False, logger=None,
                      sentence_pair_data=False, for_rnn=False):
    # TODO(SB): Simpler version for plain RNN.
    dataset = TrimDataset(dataset, seq_length, eval_mode=eval_mode, sentence_pair_data=sentence_pair_data)
    dataset = TokensToIDs(vocabulary, dataset, sentence_pair_data=sentence_pair_data)
    if for_rnn:
        dataset = CropAndPadForRNN(dataset, seq_length, logger=logger, sentence_pair_data=sentence_pair_data)
    else:
        dataset = CropAndPad(dataset, seq_length, logger=logger, sentence_pair_data=sentence_pair_data)

    if sentence_pair_data:
        X = np.transpose(np.array([[example["premise_tokens"] for example in dataset],
                      [example["hypothesis_tokens"] for example in dataset]],
                     dtype=np.int32), (1, 2, 0))
        if for_rnn: 
            # TODO(SB): Extend this clause to the non-pair case.
            transitions = np.zeros((len(dataset), 2, 0))
            num_transitions = np.zeros((len(dataset), 2))
        else:
            transitions = np.transpose(np.array([[example["premise_transitions"] for example in dataset],
                                    [example["hypothesis_transitions"] for example in dataset]],
                                   dtype=np.int32), (1, 2, 0))
            num_transitions = np.transpose(np.array(
                [[example["num_premise_transitions"] for example in dataset],
                 [example["num_hypothesis_transitions"] for example in dataset]],
                dtype=np.int32), (1, 0))
    else:
        X = np.array([example["tokens"] for example in dataset],
                     dtype=np.int32)
        transitions = np.array([example["transitions"] for example in dataset],
                               dtype=np.int32)
        num_transitions = np.array(
            [example["num_transitions"] for example in dataset],
            dtype=np.int32)
    y = np.array(
        [data_manager.LABEL_MAP[example["label"]] for example in dataset],
        dtype=np.int32)

    return X, transitions, y, num_transitions


def BuildVocabulary(raw_training_data, raw_eval_sets, embedding_path, logger=None, sentence_pair_data=False):
    # Find the set of words that occur in the data.
    logger.Log("Constructing vocabulary...")
    types_in_data = set()
    for dataset in [raw_training_data] + [eval_dataset[1] for eval_dataset in raw_eval_sets]:
        if sentence_pair_data:
            types_in_data.update(itertools.chain.from_iterable([example["premise_tokens"]
                                                                for example in dataset]))
            types_in_data.update(itertools.chain.from_iterable([example["hypothesis_tokens"]
                                                                for example in dataset]))
        else:
            types_in_data.update(itertools.chain.from_iterable([example["tokens"]
                                                                for example in dataset]))
    logger.Log("Found " + str(len(types_in_data)) + " word types.")

    if embedding_path == None:
        logger.Log(
            "Warning: Open-vocabulary models require pretrained vectors. Running with empty vocabulary.")
        vocabulary = CORE_VOCABULARY
    else:
        # Build a vocabulary of words in the data for which we have an
        # embedding.
        vocabulary = BuildVocabularyForASCIIEmbeddingFile(
            embedding_path, types_in_data, CORE_VOCABULARY)

    return vocabulary


def BuildVocabularyForASCIIEmbeddingFile(path, types_in_data, core_vocabulary):
    """Quickly iterates through a GloVe-formatted ASCII vector file to
    extract a working vocabulary of words that occur both in the data and
    in the vector file."""

    # TODO(SB): Report on *which* words are skipped. See if any are common.

    vocabulary = {}
    vocabulary.update(core_vocabulary)
    next_index = len(vocabulary)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ", 1)
            word = spl[0]
            if word in types_in_data:
                vocabulary[word] = next_index
                next_index += 1
    return vocabulary


def LoadEmbeddingsFromASCII(vocabulary, embedding_dim, path):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a GloVe - format ASCII vector file.

    For now, values not found in the file will be set to zero."""
    emb = np.zeros(
        (len(vocabulary), embedding_dim), dtype=theano.config.floatX)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ")
            word = spl[0]
            if word in vocabulary:
                emb[vocabulary[word], :] = [float(e) for e in spl[1:]]
    return emb


def TransitionsToParse(transitions, words):
    if transitions is not None:
        stack = ["(P *ZEROS*)"] * (len(transitions) + 1)
        buffer_ptr = 0
        for transition in transitions:
            if transition == 0:
                stack.append("(P " + words[buffer_ptr] +")")
                buffer_ptr += 1
            elif transition == 1:
                r = stack.pop()
                l = stack.pop()
                stack.append("(M " + l + " " + r + ")")
        return stack.pop()
    else:
        return " ".join(words)
