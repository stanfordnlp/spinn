#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )
#
# The loaded data is a dictionary with fields:
#   sentence: The raw input following the tab, i.e., the parse.
#   label: The raw label string.
#   op_sequence: An operation sequence for the shift reduce parser. Shifts are represented
#      by raw word tokens to be shifted, and reductions are marked by a special REDUCE_OP
#      token.
import itertools
import numpy as np

from rembed import util


REDUCE_OP = '*MERGE*'

NUM_CLASSES = 2


def convert_binary_bracketed_data(filename):
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            tab_split = line.split('\t')
            example["label"] = tab_split[0]
            example["sentence"] = tab_split[1]
            example["op_sequence"] = []
            for word in example["sentence"].split(' '):
                if word == ')':
                    example["op_sequence"].append(REDUCE_OP)
                elif word != '(':
                    example["op_sequence"].append(word)
            examples.append(example)
    return examples


def load_data(path, vocabulary=None, seq_length=None, batch_size=32, eval_mode=False, logger=None):
    dataset = convert_binary_bracketed_data(path)

    if not vocabulary:
        # Build vocabulary from data
        # TODO(SB): Use a fixed vocab file in case this takes especially long, or we want
        # to include vocab items that don't appear in the training data.
        vocabulary = {REDUCE_OP: -1,
                      '*PADDING*': 0}
        types = set(itertools.chain.from_iterable([example["op_sequence"]
                                                   for example in dataset]))
        types.remove(REDUCE_OP)
        vocabulary.update({type: i + 1 for i, type in enumerate(types)})

    # Convert token sequences to integer sequences
    dataset = util.tokens_to_ids(vocabulary, dataset)
    dataset = util.crop_and_pad(dataset, seq_length, logger=logger)
    X = np.array([example["op_sequence"] for example in dataset],
                 dtype=np.int32)
    y = np.array([0 if example["label"] == "F" else 1 for example in dataset],
                 dtype=np.int32)

    if logger:
        logger.Log("Loaded %i examples to sequences of length %i" %
                   (len(dataset), seq_length))

    # Build batched data iterator.
    if eval_mode:
        data_iter = util.MakeEvalIterator(X, y, batch_size)
    else:
        data_iter = util.MakeTrainingIterator(X, y, batch_size)

    return data_iter, vocabulary


if __name__ == "__main__":
    # Demo:
    examples = import_binary_bracketed_data('bl-data/bl_dev.tsv')
    print examples[0]
