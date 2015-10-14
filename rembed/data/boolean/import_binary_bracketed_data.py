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


REDUCE_OP = '*MERGE*'


def import_binary_bracketed_data(filename):
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


if __name__ == "__main__":
    # Demo:
    examples = import_binary_bracketed_data('bl_dev.tsv')
    print examples[0]
