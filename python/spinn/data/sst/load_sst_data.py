#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )

import collections
import numpy as np

from spinn import util

SENTENCE_PAIR_DATA = False

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4
}


def convert_unary_binary_bracketed_data(filename):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            if len(line) == 0:
                continue
            example["label"] = line[1]
            example["sentence"] = line
            example["tokens"] = []
            example["transitions"] = []

            words = example["sentence"].split(' ')
            for index, word in enumerate(words):
                if word[0] != "(":
                    if word == ")":  
                        # Ignore unary merges
                        if words[index - 1] == ")":
                            example["transitions"].append(1)
                    else:
                        # Downcase all words to match GloVe.
                        example["tokens"].append(word.lower())
                        example["transitions"].append(0)
            examples.append(example)
    return examples


def load_data(path, vocabulary=None, seq_length=None, batch_size=32, eval_mode=False, logger=None):
    dataset = convert_unary_binary_bracketed_data(path)
    return dataset, None


if __name__ == "__main__":
    # Demo:
    examples = import_binary_bracketed_data('sst-data/dev.txt')
    print examples[0]
