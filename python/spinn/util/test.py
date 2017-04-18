import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.misc import Args

def default_args(**kwargs):
    args = {}

    # Required Args
    args['model_dim'] = 10
    args['word_embedding_dim'] = 12
    args['vocab_size'] = 14
    args['embedding_keep_rate'] = 1.0
    args['classifier_keep_rate'] = 1.0
    args['mlp_dim'] = 16
    args['num_mlp_layers'] = 2
    args['num_classes'] = 3
    args['use_sentence_pair'] = False

    initial_embeddings = np.arange(args['vocab_size']).repeat(
        args['word_embedding_dim']).reshape(
        args['vocab_size'], -1).astype(np.float32)

    args['initial_embeddings'] = initial_embeddings

    # Tracker Args
    args['tracking_lstm_hidden_dim'] = 4
    args['transition_weight'] = None

    # Layers
    context_args = Args()
    context_args.reshape_input = lambda x, batch_size, seq_length: x
    context_args.reshape_context = lambda x, batch_size, seq_length: x
    context_args.encoder = nn.Linear(args['word_embedding_dim'], args['model_dim'])
    context_args.input_dim = args['model_dim']

    args['context_args'] = context_args

    class Reduce(nn.Module):
        def forward(self, lefts, rights, tracking):
            batch_size = len(lefts)
            return torch.chunk(torch.cat(lefts, 0) - torch.cat(rights, 0), batch_size, 0)

    composition_args = Args()
    composition_args.lateral_tracking = True
    composition_args.use_tracking_in_composition = True
    composition_args.size = args['model_dim']
    composition_args.tracker_size = args['tracking_lstm_hidden_dim']
    composition_args.transition_weight = args['transition_weight']
    composition_args.wrap_items = lambda x: torch.cat(x, 0)
    composition_args.extract_h = lambda x: x
    composition_args.extract_c = None
    composition_args.composition = Reduce()

    args['composition_args'] = composition_args

    for k in kwargs.keys():
        args[k] = kwargs[k]

    return args


def get_batch():
    X = np.array([
        [3, 1, 2, 1],
        [3, 2, 4, 5]
    ], dtype=np.int32)

    transitions = np.array([
        # First input: push a bunch onto the stack
        [0, 0, 0, 0, 1, 1, 1],
        # Second input: push, then merge, then push more. (Leaves one item
        # on the buffer.)
        [0, 0, 1, 0, 0, 1, 1]
    ], dtype=np.int32)

    return X, transitions


def get_batch_pair():
    X = np.array([
        [[3, 1, 2, 1],
         [3, 2, 4, 5]],
        [[3, 1, 2, 1],
         [3, 2, 4, 5]],
    ], dtype=np.int32).transpose((0,2,1))

    transitions = np.array([
        [[0, 0, 0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 1, 1]],
        [[0, 0, 0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 1, 1]],
    ], dtype=np.int32).transpose((0,2,1))

    return X, transitions


def MockModel(model_cls, default_args, **kwargs):
    _kwargs = default_args
    for k, v in kwargs.iteritems():
        _kwargs[k] = v
    return model_cls(**_kwargs)


def compare_models(model1, model2):
    # Check length of parameters.
    assert len(list(model1.parameters())) == len(list(model2.parameters()))

    # Check value of parameters.
    for w, _w in zip(model1.parameters(), model2.parameters()):
        assert w.size() == _w.size()
        assert all((w.data == _w.data).numpy().astype(bool).tolist())
