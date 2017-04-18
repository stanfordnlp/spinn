import numpy as np
import random
import math

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.misc import recursively_set_device


def debug_gradient(model, losses):
    model.zero_grad()

    for name, loss in losses:
        print(name)
        loss.backward(retain_variables=True)
        stats = [(p.grad.norm().data[0], p.grad.max().data[0], p.grad.min().data[0], p.size())
                 for p in model.parameters()]
        for s in stats:
            print(s)
        print

        model.zero_grad()


def reverse_tensor(var, dim):
    dim_size = var.size(dim)
    index = [i for i in range(dim_size - 1, -1, -1)]
    index = torch.LongTensor(index)
    if isinstance(var, Variable):
        index = to_gpu(Variable(index, volatile=var.volatile))
    inverted_tensor = var.index_select(dim, index)
    return inverted_tensor


def l2_cost(model, l2_lambda):
    cost = 0.0
    for w in model.parameters():
        cost += l2_lambda * torch.sum(torch.pow(w, 2))
    return cost


def flatten(l):
    if hasattr(l, '__len__'):
        return reduce(lambda x, y: x + flatten(y), l, [])
    else:
        return [l]


def the_gpu():
    return the_gpu.gpu

the_gpu.gpu = -1

def to_cuda(var, gpu):
    if gpu >= 0:
        return var.cuda()
    return var

# Chainer already has a method for moving a variable to/from GPU in-place,
# but that messes with backpropagation. So I do it with a copy. Note that
# cuda.get_device() actually returns the dummy device, not the current one
# -- but F.copy will move it to the active GPU anyway (!)
def to_gpu(var):
    return to_cuda(var, the_gpu())


def to_cpu(var):
    return to_cuda(var, -1)


def arr_to_gpu(arr):
    if the_gpu() >= 0:
        return torch.cuda.FloatTensor(arr)
    else:
        return arr


class LSTMState:
    """Class for intelligent LSTM state object.

    It can be initialized from either a tuple ``(c, h)`` or a single variable
    `both`, and provides lazy attribute access to ``c``, ``h``, and ``both``.
    Since the SPINN conducts all LSTM operations on GPU and all tensor
    shuffling on CPU, ``c`` and ``h`` are automatically moved to GPU while
    ``both`` is automatically moved to CPU.

    Args:
        inpt: Either a tuple of ~chainer.Variable objects``(c, h)`` or a single
        concatenated ~chainer.Variable containing both.

    Attributes:
        c (~chainer.Variable): LSTM memory state, moved to GPU if necessary.
        h (~chainer.Variable): LSTM hidden state, moved to GPU if necessary.
        both (~chainer.Variable): Concatenated LSTM state, moved to CPU if
            necessary.

    """
    def __init__(self, inpt):
        if isinstance(inpt, tuple):
            self._c, self._h = inpt
        else:
            self._both = inpt
            self.size = inpt.data.size()[1] // 2

    @property
    def h(self):
        if not hasattr(self, '_h'):
            self._h = to_gpu(get_h(self._both, self.size))
        return self._h

    @property
    def c(self):
        if not hasattr(self, '_c'):
            self._c = to_gpu(get_c(self._both, self.size))
        return self._c

    @property
    def both(self):
        if not hasattr(self, '_both'):
            self._both = torch.cat(
                (to_cpu(self._c), to_cpu(self._h)), 1)
        return self._both


def get_seq_h(state, hidden_dim):
    return state[:, :, hidden_dim:]

def get_seq_c(state, hidden_dim):
    return state[:, :, :hidden_dim]

def get_seq_state(c, h):
    return torch.cat([h, c], 2)


def get_h(state, hidden_dim):
    return state[:, hidden_dim:]

def get_c(state, hidden_dim):
    return state[:, :hidden_dim]

def get_state(c, h):
    return torch.cat([h, c], 1)


def bundle(lstm_iter):
    """Bundle an iterable of concatenated LSTM states into a batched LSTMState.

    Used between CPU and GPU computation. Reversed by :func:`~unbundle`.

    Args:
        lstm_iter: Iterable of ``B`` ~chainer.Variable objects, each with
            shape ``(1,2*S)``, consisting of ``c`` and ``h`` concatenated on
            axis 1.

    Returns:
        state: :class:`~LSTMState` object, with ``c`` and ``h`` attributes
            each with shape ``(B,S)``.
    """
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return LSTMState(torch.cat(lstm_iter, 0))


def unbundle(state):
    """Unbundle a batched LSTM state into a tuple of concatenated LSTM states.

    Used between GPU and CPU computation. Reversed by :func:`~bundle`.

    Args:
        state: :class:`~LSTMState` object, with ``c`` and ``h`` attributes
            each with shape ``(B,S)``, or an ``inpt`` to
            :func:`~LSTMState.__init__` that would produce such an object.

    Returns:
        lstm_iter: Iterable of ``B`` ~chainer.Variable objects, each with
            shape ``(1,2*S)``, consisting of ``c`` and ``h`` concatenated on
            axis 1.
    """
    if state is None:
        return itertools.repeat(None)
    if not isinstance(state, LSTMState):
        state = LSTMState(state)
    return torch.chunk(
        state.both, state.both.data.size()[0], 0)


def extract_gates(x, n):
    r = x.view(x.size(0), x.size(1) // n, n)
    return [r[:, :, i] for i in range(n)]


def lstm(c_prev, x):
    a, i, f, o = extract_gates(x, 4)

    a = F.tanh(a)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    o = F.sigmoid(o)

    c = a * i + f * c_prev
    h = o * F.tanh(c)

    return c, h


class ReduceTreeGRU(nn.Module):
    """
    Computes the following TreeGRU (x is optional):

    hprev = left + right
    r = sigm(Ur_x(x) + Wr(hprev))
    z = sigm(Uz_x(x) + Wz(hprev))
    c = tanh(Uc_x(x) + V_l(left * r) + V_r(right * r))
    h = (1-z) * hprev + z * c
    or:
    h = hprev + z * (c - hprev)

    Standard GRU would be:

    r = sigm(Ur_x(x) + Wr(hprev))
    z = sigm(Uz_x(x) + Wz(hprev))
    c = tanh(Uc_x(x) + V(hprev * r))
    h = (1-z) * hprev + z * c
    or:
    h = hprev + z * (c - hprev)

    """

    def __init__(self, size, tracker_size=None, use_tracking_in_composition=None):
        super(ReduceTreeGRU, self).__init__()
        self.size = size
        self.W = Linear(initializer=HeKaimingInitializer)(size, 2 * size)
        self.Vl = Linear(initializer=HeKaimingInitializer)(size, size)
        self.Vr = Linear(initializer=HeKaimingInitializer)(size, size)
        if tracker_size is not None and use_tracking_in_composition:
            self.U = Linear(initializer=HeKaimingInitializer)(tracker_size, 3 * size)

    def forward(self, left, right, tracking=None):
        def slice_gate(gate_data, hidden_dim, i):
            return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

        batch_size = len(left)
        size = self.size

        left = torch.cat(left, 0)
        right = torch.cat(right, 0)
        hprev = left + right

        W = self.W(hprev)
        r, z = [slice_gate(W, size, i) for i in range(2)]
        c = 0

        if hasattr(self, "U"):
            tracking = bundle(tracking)
            U = self.U(tracking.h)
            Ur, Uz, Uc = [slice_gate(U, size, i) for i in range(3)]
            r = Ur + r
            z = Uz + z
            c = Uc + c

        r = F.sigmoid(r)
        z = F.sigmoid(z)
        c = F.tanh(c + self.Vl(left * r) + self.Vr(right * r))
        h = hprev + z * (c - hprev)

        return torch.chunk(h, batch_size, 0)


def treelstm(c_left, c_right, gates):
    hidden_dim = c_left.size()[1]

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Compute and slice gate values
    i_gate, fl_gate, fr_gate, o_gate, cell_inp = \
        [slice_gate(gates, i) for i in range(5)]

    # Apply nonlinearities
    i_gate = F.sigmoid(i_gate)
    fl_gate = F.sigmoid(fl_gate)
    fr_gate = F.sigmoid(fr_gate)
    o_gate = F.sigmoid(o_gate)
    cell_inp = F.tanh(cell_inp)

    # Compute new cell and hidden value
    i_val = i_gate * cell_inp
    c_t = fl_gate * c_left + fr_gate * c_right + i_val
    h_t = o_gate * F.tanh(c_t)

    return (c_t, h_t)


class ModelTrainer(object):

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def save(self, filename, step, best_dev_error):
        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=-1)
            recursively_set_device(self.optimizer.state_dict(), gpu=-1)

        # Always sends Tensors to CPU.
        torch.save({
            'step': step,
            'best_dev_error': best_dev_error,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)

        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=the_gpu())
            recursively_set_device(self.optimizer.state_dict(), gpu=the_gpu())

    def load(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        # HACK: Compatability for saving supervised SPINN and loading RL SPINN.
        if 'baseline' in self.model.state_dict().keys() and 'baseline' not in model_state_dict:
            model_state_dict['baseline'] = torch.FloatTensor([0.0])

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step'], checkpoint['best_dev_error']


class Embed(nn.Module):
    def __init__(self, size, vocab_size, vectors):
        super(Embed, self).__init__()
        if vectors is None:
            self.embed = nn.Embedding(vocab_size, size)
        self.vectors = vectors

    def forward(self, tokens):
        if self.vectors is None:
            embeds = self.embed(tokens.contiguous().view(-1).long())
        else:
            embeds = self.vectors.take(tokens.data.cpu().numpy().ravel(), axis=0)
            embeds = to_gpu(Variable(torch.from_numpy(embeds), volatile=tokens.volatile))

        return embeds


class GRU(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers=1, reverse=False, bidirectional=False, dropout=None):
        super(GRU, self).__init__()
        self.model_dim = model_dim
        self.reverse = reverse
        self.bidirectional = bidirectional
        self.bi = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.rnn = nn.GRU(inp_dim, model_dim / self.bi, num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional)

    def forward(self, x, h0=None):
        bi = self.bi
        num_layers = self.num_layers
        batch_size, seq_len = x.size()[:2]
        model_dim = self.model_dim

        if self.reverse:
            x = reverse_tensor(x, dim=1)

        # Initialize state unless it is given.
        if h0 is None:
            h0 = to_gpu(Variable(torch.zeros(num_layers * bi, batch_size, model_dim / bi), volatile=not self.training))

        # Expects (input, h_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        output, hn = self.rnn(x, h0)

        if self.reverse:
            output = reverse_tensor(output, dim=1)

        return output, hn


class IntraAttention(nn.Module):
    def __init__(self, inp_size, outp_size, distance_bias=True):
        super(IntraAttention, self).__init__()
        self.inp_size = inp_size
        self.outp_size = outp_size
        self.distance_bias = distance_bias
        self.f = nn.Linear(inp_size, outp_size)

    def d(self, batch_size, seq_len, max_distance=10, scale=0.01):
        """
        Generates a bias term based on distance. Something like:

        [[[0, 1, 2, 3],
          [1, 0, 1, 2],
          [2, 1, 0, 1],
          ...
          ],
         [[0, 1, 2, 3],
          [1, 0, 1, 2],
          [2, 1, 0, 1],
          ...
          ],
          ...
         ]

        """

        bias = torch.range(0, seq_len-1).float().unsqueeze(0).repeat(seq_len, 1)
        diff = torch.range(0, seq_len-1).float().unsqueeze(1).expand_as(bias)
        bias = (bias - diff).abs()
        bias = bias.clamp(0, max_distance)

        bias = bias * scale

        bias = bias.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        bias = bias.contiguous()

        bias = to_gpu(Variable(bias, volatile=not self.training))

        return bias

    def forward(self, x):
        hidden_dim = self.outp_size
        distance_bias = self.distance_bias
        batch_size, seq_len, _ = x.size()

        f = self.f(x.view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)

        f_broadcast = f.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_dim)

        # assert f_broadcast[0, 0, 0, :] == f_broadcast[0, 1, 0, :]

        e = torch.bmm(f, f.transpose(1, 2))
        e = e.view(batch_size * seq_len, seq_len)

        if self.distance_bias:
            d = self.d(batch_size, seq_len).view(batch_size * seq_len, seq_len)
            e = e + d

        e = F.softmax(e)
        e = e.view(batch_size, seq_len, seq_len, 1)
        e = e.expand_as(f_broadcast)

        # assert e[0, 0, :, 0].sum() == 1

        a = (f_broadcast * e).sum(2).squeeze()

        return a


class EncodeGRU(GRU):
    def __init__(self, *args, **kwargs):
        super(EncodeGRU, self).__init__(*args, **kwargs)

    def forward(self, x, h0=None):
        output, hn = super(EncodeGRU, self).forward(x, h0)
        return output.contiguous()


class LSTM(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers=1, reverse=False, bidirectional=False, dropout=None):
        super(LSTM, self).__init__()
        self.model_dim = model_dim
        self.reverse = reverse
        self.bidirectional = bidirectional
        self.bi = 2 if self.bidirectional else 1
        self.num_layers = num_layers
        self.rnn = nn.LSTM(inp_dim, model_dim / self.bi, num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout)

    def forward(self, x, h0=None, c0=None):
        bi = self.bi
        num_layers = self.num_layers
        batch_size, seq_len = x.size()[:2]
        model_dim = self.model_dim

        if self.reverse:
            x = reverse_tensor(x, dim=1)

        # Initialize state unless it is given.
        if h0 is None:
            h0 = to_gpu(Variable(torch.zeros(num_layers * bi, batch_size, model_dim / bi), volatile=not self.training))
        if c0 is None:
            c0 = to_gpu(Variable(torch.zeros(num_layers * bi, batch_size, model_dim / bi), volatile=not self.training))

        # Expects (input, h_0, c_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.rnn(x, (h0, c0))

        if self.reverse:
            output = reverse_tensor(output, dim=1)

        return output


class ReduceTreeLSTM(nn.Module):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two to three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None, use_tracking_in_composition=None):
        super(ReduceTreeLSTM, self).__init__()
        self.left = Linear(initializer=HeKaimingInitializer)(size, 5 * size)
        self.right = Linear(initializer=HeKaimingInitializer)(size, 5 * size, bias=False)
        if tracker_size is not None and use_tracking_in_composition:
            self.track = Linear(initializer=HeKaimingInitializer)(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two to three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided
        as iterables and batched internally into tensors.

        Args:
            left_in: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~chainer.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left.h)
        lstm_in += self.right(right.h)
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking.h)
        out = unbundle(treelstm(left.c, right.c, lstm_in))
        return out


class MLP(nn.Module):
    def __init__(self, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_bn,
                 classifier_dropout_rate=0.0):
        super(MLP, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.mlp_bn = mlp_bn
        self.classifier_dropout_rate = classifier_dropout_rate

        features_dim = mlp_input_dim

        if mlp_bn:
            self.bn_inp = nn.BatchNorm1d(features_dim)
        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), Linear(initializer=HeKaimingInitializer)(features_dim, mlp_dim))
            if mlp_bn:
                setattr(self, 'bn{}'.format(i), nn.BatchNorm1d(mlp_dim))
            features_dim = mlp_dim
        setattr(self, 'l{}'.format(num_mlp_layers), Linear(initializer=HeKaimingInitializer)(features_dim, num_classes))

    def forward(self, h):
        if self.mlp_bn:
            h = self.bn_inp(h)
        h = F.dropout(h, self.classifier_dropout_rate, training=self.training)
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if self.mlp_bn:
                bn = getattr(self, 'bn{}'.format(i))
                h = bn(h)
            h = F.dropout(h, self.classifier_dropout_rate, training=self.training)
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        y = layer(h)
        return y


class HeKaimingLinear(nn.Linear):
    def reset_parameters(self):
        HeKaimingInitializer(self.weight)
        if self.bias is not None:
            ZeroInitializer(self.bias)


def DefaultUniformInitializer(param):
    stdv = 1. / math.sqrt(param.size(1))
    UniformInitializer(param, stdv)


def HeKaimingInitializer(param):
    fan = param.size()
    init = np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
                                size=fan).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def UniformInitializer(param, range):
    shape = param.size()
    init = np.random.uniform(-range, range, shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def NormalInitializer(param, std):
    shape = param.size()
    init = np.random.normal(0.0, std, shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def OneInitializer(param):
    shape = param.size()
    init = np.ones(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def ValueInitializer(param, value):
    shape = param.size()
    init = np.ones(shape).astype(np.float32) * value
    param.data.set_(torch.from_numpy(init))


def TreeLSTMBiasInitializer(param):
    shape = param.size()

    hidden_dim = shape[0] / 5
    init = np.zeros(shape).astype(np.float32)
    init[hidden_dim:3*hidden_dim] = 1

    param.data.set_(torch.from_numpy(init))


def LSTMBiasInitializer(param):
    shape = param.size()

    hidden_dim = shape[0] / 4
    init = np.zeros(shape)
    init[hidden_dim:2*hidden_dim] = 1

    param.data.set_(torch.from_numpy(init))


def DoubleIdentityInitializer(param, range):
    shape = param.size()

    half_d = shape[0] / 2
    double_identity = np.concatenate((
        np.identity(half_d), np.identity(half_d))).astype(np.float32)

    param.data.set_(torch.from_numpy(double_identity)).add_(
        UniformInitializer(param.clone(), range))


def PassthroughLSTMInitializer(lstm):
    G_POSITION = 3
    F_POSITION = 1

    i_dim = lstm.weight_ih_l0.size()[1]
    h_dim = lstm.weight_hh_l0.size()[1]
    assert i_dim == h_dim, "PassthroughLSTM requires input dim == hidden dim."

    hh_init = np.zeros(lstm.weight_hh_l0.size()).astype(np.float32)
    ih_init = np.zeros(lstm.weight_ih_l0.size()).astype(np.float32)
    ih_init[G_POSITION * h_dim:(G_POSITION + 1) * h_dim, :] = np.identity(h_dim)

    bhh_init = np.zeros(lstm.bias_hh_l0.size()).astype(np.float32)
    bih_init = np.ones(lstm.bias_ih_l0.size()).astype(np.float32) * 2
    bih_init[G_POSITION * h_dim:(G_POSITION + 1) * h_dim] = 0
    bih_init[F_POSITION * h_dim:(F_POSITION + 1) * h_dim] = -2

    lstm.weight_hh_l0.data.set_(torch.from_numpy(hh_init))
    lstm.weight_ih_l0.data.set_(torch.from_numpy(ih_init))
    lstm.bias_hh_l0.data.set_(torch.from_numpy(bhh_init))
    lstm.bias_ih_l0.data.set_(torch.from_numpy(bih_init))


def Linear(initializer=DefaultUniformInitializer, bias_initializer=ZeroInitializer):
    class CustomLinear(nn.Linear):
        def reset_parameters(self):
            initializer(self.weight)
            if self.bias is not None:
                bias_initializer(self.bias)
    return CustomLinear
