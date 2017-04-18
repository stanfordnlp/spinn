#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )

import collections
import numpy as np
import sys

from spinn import util
from spinn.data.sst.base import convert_unary_binary_bracketed_data

SENTENCE_PAIR_DATA = False

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4
}


def load_data(path, vocabulary=None, seq_length=None, batch_size=32, eval_mode=False, logger=None):
    dataset = convert_unary_binary_bracketed_data(path)
    return dataset, None


if __name__ == "__main__":
    pass
