
from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T


def UniformInitializer(range):
    return lambda shape: np.random.uniform(-range, range, shape)


def NormalInitializer(std):
    return lambda shape: np.random.normal(0.0, std, shape)


def ZeroInitializer():
    return lambda shape: np.zeros(shape)


class VariableStore(object):

    def __init__(self, prefix="vs", default_initializer=UniformInitializer(0.1)):
        self.prefix = prefix
        self.default_initializer = default_initializer
        self.vars = {}

    def add_param(self, name, shape, initializer=None):
        if not initializer:
            initializer = self.default_initializer

        if name not in self.vars:
            full_name = "%s/%s" % (self.prefix, name)
            print "Created variable " + full_name
            self.vars[name] = theano.shared(initializer(shape),
                                            name=full_name)
        return self.vars[name]


def ReLULayer(inp, inp_dim, outp_dim, vs, name="tanh_layer", use_bias=True):
    pre_nl = Linear(inp, inp_dim, outp_dim, vs, name, use_bias)
    # ReLU isn't present in this version of Theano.
    outp = T.maximum(pre_nl, 0)

    return outp


def Linear(inp, inp_dim, outp_dim, vs, name="linear", use_bias=True):
    W = vs.add_param("%s_W" % name, (inp_dim, outp_dim))
    outp = inp.dot(W)

    if use_bias:
        b = vs.add_param("%s_b" % name, (outp_dim,),
                         initializer=ZeroInitializer())
        outp += b

    return outp


def SGD(cost, params, lr=0.01):
    grads = T.grad(cost, params)

    new_values = OrderedDict()
    for param, grad in zip(params, grads):
        new_values[param] = param - lr * grad

    return new_values


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
    # Stolen from Lasagne source code
    # https://raw.githubusercontent.com/Lasagne/Lasagne/master/lasagne/utils.py
    """
    Helper function to unroll for loops. Can be used to unroll theano.scan.
    The parameter names are identical to theano.scan, please refer to here
    for more information.

    Note that this function does not support the truncate_gradient
    setting from theano.scan.

    Parameters
    ----------

    fn : function
        Function that defines calculations at each step.

    sequences : TensorVariable or list of TensorVariables
        List of TensorVariable with sequence data. The function iterates
        over the first dimension of each TensorVariable.

    outputs_info : list of TensorVariables
        List of tensors specifying the initial values for each recurrent
        value.

    non_sequences: list of TensorVariables
        List of theano.shared variables that are used in the step function.

    n_steps: int
        Number of steps to unroll.

    go_backwards: bool
        If true the recursion starts at sequences[-1] and iterates
        backwards.

    Returns
    -------
    List of TensorVariables. Each element in the list gives the recurrent
    values at each time step.

    """
    if not isinstance(sequences, (list, tuple)):
        sequences = [sequences]

    # When backwards reverse the recursion direction
    counter = range(n_steps)
    if go_backwards:
        counter = counter[::-1]

    output = []
    prev_vals = outputs_info
    for i in counter:
        step_input = [s[i] for s in sequences] + prev_vals + non_sequences
        out_ = fn(*step_input)
        # The returned values from step can be either a TensorVariable,
        # a list, or a tuple.  Below, we force it to always be a list.
        if isinstance(out_, T.TensorVariable):
            out_ = [out_]
        if isinstance(out_, tuple):
            out_ = list(out_)
        output.append(out_)

        prev_vals = output[-1]

    # iterate over each scan output and convert it to same format as scan:
    # [[output11, output12,...output1n],
    # [output21, output22,...output2n],...]
    output_scan = []
    for i in range(len(output[0])):
        l = map(lambda x: x[i], output)
        output_scan.append(T.stack(*l))

    return output_scan
