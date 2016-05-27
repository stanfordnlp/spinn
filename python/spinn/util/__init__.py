from collections import namedtuple

from theano.sandbox.cuda import cuda_available

# Only import custom CUDA ops if we on a CUDA-enabled host.
if cuda_available:
    from spinn.util.cuda import *

from spinn.util.theano_internal import *
from spinn.util.data import *
from spinn.util.blocks import *
from spinn.util.variable_store import VariableStore


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


