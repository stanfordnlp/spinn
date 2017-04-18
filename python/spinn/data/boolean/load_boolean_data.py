
#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )
import numpy as np

from spinn import util

NUM_CLASSES = 2

SENTENCE_PAIR_DATA = False

FIXED_VOCABULARY = {
    util.PADDING_TOKEN: 0,
    "T": 1,
    "F": 2,
    "not": 3,
    "and": 4,
    "or": 5
}

LABEL_MAP = {
    "T": 0,
    "F": 1
}


def convert_binary_bracketed_data(filename):
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            tab_split = line.split('\t')
            example["label"] = tab_split[0]
            example["sentence"] = tab_split[1]
            example["tokens"] = []
            example["transitions"] = []

            for word in example["sentence"].split(' '):
                if word != "(":
                    if word != ")":
                        example["tokens"].append(word)
                    example["transitions"].append(1 if word == ")" else 0)

            example["example_id"] = str(len(examples))

            examples.append(example)
    return examples


def load_data(path):
    dataset = convert_binary_bracketed_data(path)
    return dataset, FIXED_VOCABULARY

if __name__ == "__main__":
    # Demo:
    examples = import_binary_bracketed_data('bl-data/bl_dev.tsv')
    print examples[0]
