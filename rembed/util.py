from collections import OrderedDict
import random

import numpy as np
import theano
from theano import tensor as T


def UniformInitializer(range):
    return lambda shape: np.random.uniform(-range, range, shape)


def NormalInitializer(std):
    return lambda shape: np.random.normal(0.0, std, shape)


def ZeroInitializer():
    return lambda shape: np.zeros(shape)


def DoubleIdentityInitializer(range):
    def init(shape):
        half_d = shape[0] / 2
        double_identity = np.concatenate((
            np.identity(half_d), np.identity(half_d)))
        return double_identity + UniformInitializer(range)(shape)
    return init


class VariableStore(object):

    def __init__(self, prefix="vs", default_initializer=UniformInitializer(0.1), logger=None):
        self.prefix = prefix
        self.default_initializer = default_initializer
        self.vars = {}
        self.logger = logger

    def add_param(self, name, shape, initializer=None):
        if not initializer:
            initializer = self.default_initializer

        if name not in self.vars:
            full_name = "%s/%s" % (self.prefix, name)
            if self.logger:
                self.logger.Log(
                    "Created variable " + full_name, level=self.logger.DEBUG)
            self.vars[name] = theano.shared(initializer(shape),
                                            name=full_name)
        return self.vars[name]


def ReLULayer(inp, inp_dim, outp_dim, vs, name="relu_layer", use_bias=True, initializer=None):
    pre_nl = Linear(inp, inp_dim, outp_dim, vs, name, use_bias, initializer)
    # ReLU isn't present in this version of Theano.
    outp = T.maximum(pre_nl, 0)

    return outp


def Linear(inp, inp_dim, outp_dim, vs, name="linear_layer", use_bias=True, initializer=None):
    W = vs.add_param("%s_W" %
                     name, (inp_dim, outp_dim), initializer=initializer)
    outp = inp.dot(W)

    if use_bias:
        b = vs.add_param("%s_b" % name, (outp_dim,),
                         initializer=ZeroInitializer())
        outp += b

    return outp


def LSTM(lstm_prev, inp, inp_dim, hidden_dim, vs, name="lstm"):
    # input -> hidden mapping
    W = vs.add_param("%s/W" % name, (inp_dim, hidden_dim * 4))
    # hidden -> hidden mapping
    U = vs.add_param("%s/U" % name, (hidden_dim, hidden_dim * 4))
    # gate biases
    # TODO(jgauthier): support excluding params from regularization
    b = vs.add_param("%s/b" % name, (hidden_dim * 4,),
                     initializer=ZeroInitializer())

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Decompose previous LSTM value into hidden and cell value
    h_prev = lstm_prev[:, :hidden_dim]
    c_prev = lstm_prev[:, hidden_dim:]

    # Compute and slice gate values
    gates = T.dot(inp, W) + T.dot(h_prev, U) + b
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
        name="mlp"):
    if hidden_dims is None:
        hidden_dims = []

    prev_val = inp
    dims = [inp_dim] + hidden_dims + [outp_dim]
    for i, (src_dim, tgt_dim) in enumerate(zip(dims, dims[1:])):
        prev_val = layer(prev_val, src_dim, tgt_dim, vs, use_bias=True,
                         name="%s/%i" % (name, i))
    return prev_val


def SGD(cost, params, lr=0.01):
    grads = T.grad(cost, params)

    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        new_values[param] = param - lr * grad

    return new_values


def embedding_SGD(cost, embedding_matrix, lr=0.01, used_embeddings=None):
    new_values = OrderedDict()

    if used_embeddings:
        grads = T.grad(cost, wrt=used_embeddings)
        new_value = (used_embeddings,
                     T.inc_subtensor(used_embeddings, -lr * grads))
    else:
        new_values = SGD(cost, [embedding_matrix], lr)
        new_value = (embedding_matrix, new_values[embedding_matrix])

    return new_value


def momentum(cost, params, lr=0.01, momentum=0.9):
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


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    # From:
    # https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def trim_dataset(dataset, seq_length, eval_mode=False):
    """Avoid using excessively long training examples."""
    if eval_mode:
        return dataset
    else:
        new_dataset = [example for example in dataset if len(
            example["transitions"]) <= seq_length]
        return new_dataset


def tokens_to_ids(vocabulary, dataset):
    """Replace strings in original boolean dataset with token IDs."""

    unk_id = vocabulary["*UNK*"]
    for example in dataset:
        example["tokens"] = [vocabulary.get(token, unk_id)
                             for token in example["tokens"]]
    return dataset


def crop_and_pad_example(example, left_padding, target_length, key, logger=None):
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


def crop_and_pad(dataset, length, logger=None):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    # Always make sure that the transitions are aligned at the left edge, so
    # the final stack top is the root of the tree. If cropping is used, it should
    # just introduce empty nodes into the tree.
    for example in dataset:
        transitions_left_padding = length - len(example["transitions"])
        shifts_before_crop_and_pad = example["transitions"].count(0)
        crop_and_pad_example(
            example, transitions_left_padding, length, "transitions", logger=logger)
        shifts_after_crop_and_pad = example["transitions"].count(0)
        tokens_left_padding = shifts_after_crop_and_pad - \
            shifts_before_crop_and_pad
        crop_and_pad_example(
            example, tokens_left_padding, length, "tokens", logger=logger)
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
            if start > dataset_size:
                print "Epoch."
                # Start another epoch
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            yield tuple(source[batch_indices] for source in sources)
    return data_iter()


def MakeEvalIterator(sources, batch_size):
    # Make a list of minibatches from a dataset to use as an iterator.
    # TODO(SB): Handle the last few examples in the eval set if they don't
    # form a batch.

    dataset_size = len(sources[0])
    data_iter = []
    start = -batch_size
    while True:
        start += batch_size
        if start > dataset_size:
            break
        data_iter.append(tuple(source[start:start + batch_size]
                               for source in sources))
    return data_iter
