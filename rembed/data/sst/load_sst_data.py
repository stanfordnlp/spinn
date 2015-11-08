#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )

import collections
import numpy as np

from rembed import util

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4
}


def convert_binary_bracketed_data(filename):
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

            for word in example["sentence"].split(' '):
                if word[0] != "(":
                    if word == ")":
                        example["transitions"].append(1)
                    else:
                        # Downcase all words to match GloVe.
                        example["tokens"].append(word.lower())
                        example["transitions"].append(0)
            examples.append(example)
    return examples


def load_data(path, vocabulary=None, seq_length=None, batch_size=32, eval_mode=False, logger=None):
    dataset = convert_binary_bracketed_data(path)
    return dataset, None


if __name__ == "__main__":
    # Demo:
    examples = import_binary_bracketed_data('sst-data/dev.txt')
    print examples[0]
