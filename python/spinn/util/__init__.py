from collections import namedtuple


from spinn.util.data import *


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


