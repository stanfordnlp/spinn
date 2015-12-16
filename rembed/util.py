from collections import OrderedDict
import itertools
import random

import numpy as np
import tensorflow as tf

numpy_random = np.random.RandomState(1234)
tf.set_random_seed(numpy_random.randint(999999))

# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "*PADDING*"

# Temporary hack: Map UNK to "_" when loading pretrained embedding matrices:
# it's a common token that is pretrained, but shouldn't look like any content words.
UNK_TOKEN = "_"

CORE_VOCABULARY = {PADDING_TOKEN: 0,
                   UNK_TOKEN: 1}


def UniformInitializer(range):
    return lambda shape: np.random.uniform(-range, range, shape)


def NormalInitializer(std):
    return lambda shape: np.random.normal(0.0, std, shape)


def ZeroInitializer():
    return lambda shape: np.zeros(shape)


def TreeLSTMBiasInitializer():
    def init(shape):
        hidden_dim = shape[0] / 5
        value = np.zeros(shape)
        value[hidden_dim:3*hidden_dim] = 1
        return value


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
            init_value = initializer(shape).astype(np.float32)
            self.vars[name] = tf.Variable(init_value, name=full_name)
        return self.vars[name]


def ReLULayer(inp, inp_dim, outp_dim, vs, name="relu_layer", use_bias=True, initializer=None):
    pre_nl = Linear(inp, inp_dim, outp_dim, vs, name, use_bias, initializer)
    # ReLU isn't present in this version of Theano.
    outp = tf.nn.relu(pre_nl)

    return outp


def Linear(inp, inp_dim, outp_dim, vs, name="linear_layer", use_bias=True, initializer=None):
    W = vs.add_param("%s_W" %
                     name, (inp_dim, outp_dim), initializer=initializer)
    outp = tf.matmul(inp, W)

    if use_bias:
        b = vs.add_param("%s_b" % name, (outp_dim,),
                         initializer=ZeroInitializer())
        outp += b

    return outp


def Dropout(inp, keep_rate, apply_dropout):
    """Apply dropout to a set of activations.

    Args:
      inp: Input vector.
      keep_rate: Dropout parameter. 1.0 entails no dropout.
      apply_dropout: A Theano scalar indicating whether to apply dropout (1.0)
        or eval-mode rescaling (0.0).
    """
    # TODO(SB): Investigate whether a Theano conditional would be faster than the linear combination below.

    dropout_candidate = tf.nn.dropout(inp, keep_rate)
    rescaling_candidate = keep_rate * inp
    result = apply_dropout * dropout_candidate + (1 - apply_dropout) * rescaling_candidate

    return result



def IdentityLayer(inp, inp_dim, outp_dim, vs, name="identity_layer", use_bias=True, initializer=None):
    """An identity function that takes the same parameters as the above layers."""
    assert inp_dim == outp_dim, "Identity layer requires inp_dim == outp_dim."
    return inp


def TreeLSTMLayer(lstm_prev, _, full_memory_dim, vs, name="tree_lstm", initializer=None):
    assert full_memory_dim % 2 == 0, "Input is concatenated (h, c); dim must be even."
    hidden_dim = full_memory_dim / 2

    W = vs.add_param("%s/W" % name, (hidden_dim * 2, hidden_dim * 5),
                     initializer=initializer)
    b = vs.add_param("%s/b" % name, (hidden_dim * 5,),
                     initializer=TreeLSTMBiasInitializer())

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Decompose previous LSTM value into hidden and cell value
    l_h_prev = lstm_prev[:, :hidden_dim]
    l_c_prev = lstm_prev[:, hidden_dim:2 * hidden_dim]
    r_h_prev = lstm_prev[:, 2 * hidden_dim:3 * hidden_dim]
    r_c_prev = lstm_prev[:, 3 * hidden_dim:]

    h_prev = T.concatenate([l_h_prev, r_h_prev], axis=1)

    # Compute and slice gate values
    gates = T.dot(h_prev, W) + b
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


def PreprocessDataset(dataset, vocabulary, seq_length, data_manager, eval_mode=False, logger=None,
                      sentence_pair_data=False):
    dataset = TrimDataset(dataset, seq_length, eval_mode=eval_mode, sentence_pair_data=sentence_pair_data)
    dataset = TokensToIDs(vocabulary, dataset, sentence_pair_data=sentence_pair_data)
    dataset = CropAndPad(dataset, seq_length, logger=logger, sentence_pair_data=sentence_pair_data)

    if sentence_pair_data:
        X = np.transpose(np.array([[example["premise_tokens"] for example in dataset],
                      [example["hypothesis_tokens"] for example in dataset]],
                     dtype=np.int32), (1, 2, 0))
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


def convert_labels_to_onehot(labels, batch_size, num_classes):
    """
    Convert a vector of integer class labels to a matrix of one-hot target vectors.
    """
    with tf.name_scope("onehot"):
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        sparse_ptrs = tf.concat(1, [indices, labels], name="ptrs")
        onehots = tf.sparse_to_dense(sparse_ptrs, [batch_size, num_classes],
                                     1.0, 0.0)
        return onehots


@tf.RegisterGradient("ScatterUpdate")
def _ScatterUpdateGrad(op, grad):
    assert len(op.inputs) == 3
    _, _, updates = op.inputs
    return [None, None, tf.ones_like(updates)]
