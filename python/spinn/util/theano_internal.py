"""Low-level Theano utilities."""

from collections import OrderedDict
from functools import wraps

import theano
from theano import ifelse
from theano import tensor as T
from theano.compile.sharedvalue import SharedVariable
from theano.sandbox.cuda import cuda_available

if cuda_available:
    from theano.sandbox.cuda import HostFromGpu
    from spinn.util import cuda


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


def merge_updates(*updates_dicts):
    all_updates = OrderedDict()
    for updates_dict in updates_dicts:
        for k, v in updates_dict.iteritems():
            if k in all_updates:
                all_updates[k] += v
            else:
                all_updates[k] = v

    return all_updates


def batch_subgraph_gradients(g_in, wrt, f_g_out,
                             wrt_jacobian=True,
                             name="batch_subgraph_grad"):
    """
    Build gradients w.r.t. some cost on a subgraph of a larger graph.

    Let G be a feedforward subgraph for which we want to compute gradients.
    G has N_I inputs and N_O outputs.

    This function will compute batch gradients on the subgraph G.

    It optionally supports computing Jacobians (batch-element-wise cost
    gradients) as well, though this is experimental and relies on some naughty
    Theano hacks.

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
        d_wrt = map(cuda.strip_transfer, d_wrt)
        d_in = map(cuda.strip_transfer, d_in)
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
